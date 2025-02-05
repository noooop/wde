from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from wde.logger import init_logger
from wde.utils import lazy_import
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.decoding.kv_cache.logic_manager import LogicKVCacheManager
from wde.workflows.decoding.kv_cache.naive.scheduler import \
    DecodingSchedulingBudget
from wde.workflows.decoding.kv_cache.offloading.manager import \
    OffloadingManager
from wde.workflows.decoding.kv_cache.prefix_caching.scheduler import \
    PrefixCachingDecodingScheduler
from wde.workflows.decoding.schema.engine_io import DecodingSchedulerOutput
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

logger = init_logger(__name__)


@dataclass
class DecodingSwapInSchedulingBudget:
    max_num_swap_in_blocks: int
    max_num_swap_in_requests: int
    _num_curr_blocks: int = 0
    _num_curr_requests: int = 0
    need_swap_in_blocks: List = field(default_factory=list)

    @classmethod
    def from_engine_config(cls, engine_config):
        return cls(
            max_num_swap_in_blocks=engine_config.scheduler_config.
            max_num_swap_in_blocks,
            max_num_swap_in_requests=engine_config.scheduler_config.
            max_num_swap_in_requests,
        )

    def full(self) -> bool:
        if self._num_curr_blocks >= self.max_num_swap_in_blocks:
            return True

        if self._num_curr_requests >= self.max_num_swap_in_requests:
            return True

        return False

    def remaining_blocks_budget(self):
        return self.max_num_swap_in_blocks - self._num_curr_blocks

    def add_num_requests(self, req_id: str, num_curr_requests: int):
        self._num_curr_requests += num_curr_requests

    def add_num_blocks(self, req_id: str, num_blocks: int):
        self._num_curr_blocks += num_blocks


@dataclass
class SchedulerSwapInRunningOutputs:
    busy_requests: List[DecodingSchedulableRequest]
    running_queue: Deque[DecodingSchedulableRequest]


class OffloadingKVCachingDecodingScheduler(PrefixCachingDecodingScheduler):
    name = "Offloading KV Caching"
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(self, engine_config: EngineConfig,
                 request_processor: RequestProcessor, kv_cache_manager,
                 offloading_manager: OffloadingManager) -> None:
        super().__init__(engine_config, request_processor, kv_cache_manager)
        self.offloading_manager = offloading_manager

    @classmethod
    def from_engine(cls, engine):
        block_allocator_class = lazy_import(engine.workflow.BlockAllocator)

        gpu_kv_cache_manager = LogicKVCacheManager.from_engine(
            engine=engine, block_allocator_class=block_allocator_class)

        cpu_cache = engine.kv_cache_manager.cpu_cache
        gpu_cache = engine.kv_cache_manager.gpu_cache

        offloading_manager = OffloadingManager(
            engine_config=engine.engine_config,
            cpu_cache=cpu_cache,
            gpu_cache=gpu_cache,
            gpu_block_allocator=gpu_kv_cache_manager.block_allocator)
        return cls(engine.engine_config,
                   engine.request_processor,
                   kv_cache_manager=gpu_kv_cache_manager,
                   offloading_manager=offloading_manager)

    def schedule(self) -> Optional[DecodingSchedulerOutput]:
        self.offloading_manager.check_finishd_task()

        scheduler_outputs = super().schedule()

        swap_in_task = self.offloading_manager.get_swap_in_task(
            scheduler_outputs)

        if swap_in_task is not None and scheduler_outputs.is_empty():
            swap_in_task.wait()

        scheduler_outputs.swap_out_task = self.offloading_manager.get_swap_out_task(
            scheduler_outputs)

        return scheduler_outputs

    def _schedule(self) -> DecodingSchedulerOutput:
        if not self.waiting and not self.running:
            return DecodingSchedulerOutput.create_empty()

        # schedule waiting
        waiting_scheduled = self._schedule_waiting()
        self.running.extend(waiting_scheduled.scheduled_requests)

        # schedule swap_in
        swap_in_budget = DecodingSwapInSchedulingBudget.from_engine_config(
            self.engine_config)

        swap_in_scheduled = self._schedule_swap_in_runnings(swap_in_budget)
        running_queue = swap_in_scheduled.running_queue

        # schedule running
        budget = DecodingSchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        running_scheduled = self._schedule_running(budget, running_queue)

        self.running = running_queue
        self.running.extend(running_scheduled.decode_requests)
        self.running.extend(running_scheduled.prefill_requests)
        self.running.extend(swap_in_scheduled.busy_requests)
        self.running.extend(running_scheduled.preempted)

        scheduled_requests = []
        scheduled_requests.extend(running_scheduled.decode_requests)
        scheduled_requests.extend(running_scheduled.prefill_requests)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_requests <= self.scheduler_config.max_num_requests

        return DecodingSchedulerOutput(
            scheduled_requests=scheduled_requests,
            num_batched_tokens=budget.num_batched_tokens,
            num_requests=budget.num_curr_requests,
            ignored_requests=waiting_scheduled.ignored_requests,
            need_swap_in_blocks=swap_in_budget.need_swap_in_blocks)

    def _schedule_swap_in_runnings(self, swap_in_budget):
        busy_requests = []
        running_queue = []

        if not self.running:
            return SchedulerSwapInRunningOutputs(
                busy_requests=busy_requests,
                running_queue=deque(running_queue))

        requests, busy_requests = self._filter_out_busy_requests()

        running_queue = deque()

        while requests:
            request = requests.popleft()

            # 1. Write new token ids to vblock & try to hit prefix caching
            self.kv_cache_manager.update(request)

            # 2. Check if hit prefix caching ready
            if not request.vblock.ready():
                busy_requests.append(request)
                continue

            # 3. find blocks need swap in
            curr_need_swap_in_blocks = self.offloading_manager.get_swap_in_blocks(
                request)

            if not curr_need_swap_in_blocks:
                # no need swap in
                running_queue.append(request)
                continue

            # following code schedule swap_in
            # 5. anyway, need swap in request should not goto running_queue
            busy_requests.append(request)

            # 6. test kv_cache is above high_watermark
            if self.kv_cache_manager.high_watermark():
                continue

            # 7. meet swap in budget
            num_need_swap_in_blocks = len(curr_need_swap_in_blocks)

            budget_bound_num_blocks = min(
                num_need_swap_in_blocks,
                swap_in_budget.remaining_blocks_budget())

            if budget_bound_num_blocks == 0:
                continue

            curr_need_swap_in_blocks = curr_need_swap_in_blocks[:
                                                                budget_bound_num_blocks]

            # 8. try to allocate & lock blocks avoid recompute
            allocated_swap_in_blocks = self.offloading_manager.try_allocate_swap_in_blocks(
                curr_need_swap_in_blocks)

            # 9. add to swap_in_task
            num_blocks = len(allocated_swap_in_blocks)
            swap_in_budget.need_swap_in_blocks.extend(allocated_swap_in_blocks)

            # 10. add to budget
            swap_in_budget.add_num_blocks(request.request_id, num_blocks)
            swap_in_budget.add_num_requests(request.request_id, 1)

        return SchedulerSwapInRunningOutputs(busy_requests=busy_requests,
                                             running_queue=running_queue)

    def join(self):
        super().join()
        self.offloading_manager.join()
