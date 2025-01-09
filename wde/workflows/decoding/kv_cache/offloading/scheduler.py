import time
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional

from wde.logger import init_logger
from wde.utils import lazy_import
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.decoding.kv_cache.logic_manager import (LogicKVCacheManager,
                                                           NoFreeBlocksError)
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
class SchedulerSwapInPrefillsOutputs:
    ignored_requests: List[DecodingSchedulableRequest]
    all_swap_in_able_swap_in_ed: List[DecodingSchedulableRequest]


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

        if self.record_metrics:
            scheduling_begin_ts = time.perf_counter()

        scheduler_outputs = self._schedule()

        for request in scheduler_outputs.scheduled_requests:
            request.busy = True

        swap_in_task = self.offloading_manager.get_swap_in_task(
            scheduler_outputs)

        if swap_in_task is not None and scheduler_outputs.is_empty():
            swap_in_task.wait()

        scheduler_outputs.swap_out_task = self.offloading_manager.get_swap_out_task(
            scheduler_outputs)

        if self.record_metrics:
            scheduling_end_ts = time.perf_counter()
            scheduling_time = scheduling_end_ts - scheduling_begin_ts
            num_requests = scheduler_outputs.num_requests
            num_batched_tokens = scheduler_outputs.num_batched_tokens
            for request in scheduler_outputs.scheduled_requests:
                request.metrics.scheduling_time = scheduling_time
                request.metrics.num_requests = num_requests
                request.metrics.num_batched_tokens = num_batched_tokens

        return scheduler_outputs

    def _schedule(self) -> DecodingSchedulerOutput:
        swap_in_budget = DecodingSwapInSchedulingBudget.from_engine_config(
            self.engine_config)

        swap_in_prefills = self._schedule_swap_in_prefills(swap_in_budget)
        self.running.extend(swap_in_prefills.all_swap_in_able_swap_in_ed)

        scheduled_requests = []

        budget = DecodingSchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        if self.running:
            running_queue, busy_requests = self._split_running_queue()

            if running_queue:
                self._schedule_in_runnings(running_queue, swap_in_budget)

                running_scheduled = self._schedule_running(
                    budget, running_queue)

                self.running = running_queue

                self.running.extend(running_scheduled.decode_requests)
                self.running.extend(running_scheduled.prefill_requests)

                self.running.extend(busy_requests)
                self.running.extend(running_scheduled.preempted)

                scheduled_requests.extend(running_scheduled.decode_requests)
                scheduled_requests.extend(running_scheduled.prefill_requests)

            else:
                self.running = deque(busy_requests)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_requests <= self.scheduler_config.max_num_requests

        return DecodingSchedulerOutput(
            scheduled_requests=scheduled_requests,
            num_batched_tokens=budget.num_batched_tokens,
            num_requests=budget.num_curr_requests,
            ignored_requests=swap_in_prefills.ignored_requests,
            need_swap_in_blocks=swap_in_budget.need_swap_in_blocks)

    def _schedule_swap_in_prefills(self, swap_in_budget):
        ignored_requests: List[DecodingSchedulableRequest] = []
        all_swap_in_able_swap_in_ed: List[DecodingSchedulableRequest] = []

        waiting_queue = self.waiting
        while waiting_queue:
            if swap_in_budget.full():
                break

            if self.kv_cache_manager.high_watermark():
                break

            request = waiting_queue[0]

            # 1. Check if it has been aborted
            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                waiting_queue.popleft()
                continue

            # 2. Check if request been preprocessored
            if not isinstance(request, DecodingSchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            # The busy flag now acts as whether all_swap_in_able_swap_in_ed
            if not request.busy:
                # 3. Check if the input exceeds the maximum length
                num_prompt_token_ids = request.num_prompt_token_ids
                prompt_limit = self.scheduler_config.max_model_len
                if num_prompt_token_ids > prompt_limit:
                    logger.warning(
                        "Input prompt (%d tokens) is too long"
                        " and exceeds limit of %d", num_prompt_token_ids,
                        prompt_limit)

                    ignored_requests.append(request)
                    waiting_queue.popleft()
                    continue

                # 4. create vblock &
                if request.vblock is None:
                    self.kv_cache_manager.create_vblock(request)

                # 5. try to hit prefix caching
                self.kv_cache_manager.update(request)

            # 6. find blocks need swap in
            curr_need_swap_in_blocks = self.offloading_manager.get_swap_in_blocks(
                request)

            if not curr_need_swap_in_blocks:
                waiting_queue.popleft()
                swap_in_budget.add_num_requests(request.request_id, 1)
                all_swap_in_able_swap_in_ed.append(request)
                continue

            # 7. meet budget
            num_need_swap_in_blocks = len(curr_need_swap_in_blocks)

            budget_bound_num_blocks = min(
                num_need_swap_in_blocks,
                swap_in_budget.remaining_blocks_budget())

            # 8. lock all block avoid recompute
            curr_need_swap_in_blocks = curr_need_swap_in_blocks[:
                                                                budget_bound_num_blocks]

            # 9. try allocate
            allocated_swap_in_blocks = []
            for cpu_block, gpu_block in curr_need_swap_in_blocks:
                assert cpu_block.ready()
                # read from cpu_block
                cpu_block.incr()

                assert gpu_block.ready()
                # write to gpu_block, need acquire lock

                try:
                    self.offloading_manager.gpu_block_allocator.allocate_block(
                        gpu_block)
                except NoFreeBlocksError:
                    break

                gpu_block.acquire()
                allocated_swap_in_blocks.append((cpu_block, gpu_block))

            # 10. add to swap_in_task
            num_blocks = len(allocated_swap_in_blocks)
            swap_in_budget.need_swap_in_blocks.extend(allocated_swap_in_blocks)

            # 11. add to budget
            swap_in_budget.add_num_blocks(request.request_id, num_blocks)
            swap_in_budget.add_num_requests(request.request_id, 1)

            if num_need_swap_in_blocks == num_blocks:
                request.busy = False
                waiting_queue.popleft()
                all_swap_in_able_swap_in_ed.append(request)
            else:
                # The busy flag now acts as whether all_swap_in_able_swap_in_ed
                request.busy = True

        return SchedulerSwapInPrefillsOutputs(
            ignored_requests=ignored_requests,
            all_swap_in_able_swap_in_ed=all_swap_in_able_swap_in_ed)

    def _schedule_in_runnings(self, running_queue, swap_in_budget):
        for request in running_queue:
            if swap_in_budget.full():
                break

            if self.kv_cache_manager.high_watermark():
                break

            # 1. Write new token ids to vblock & try to hit prefix caching
            self.kv_cache_manager.update(request)

            # 2. find blocks need swap in
            curr_need_swap_in_blocks = self.offloading_manager.get_swap_in_blocks(
                request)

            if not curr_need_swap_in_blocks:
                swap_in_budget.add_num_requests(request.request_id, 1)
                continue

            # 3. meet budget
            num_need_swap_in_blocks = len(curr_need_swap_in_blocks)

            budget_bound_num_blocks = min(
                num_need_swap_in_blocks,
                swap_in_budget.remaining_blocks_budget())

            # 4. lock all block avoid recompute
            curr_need_swap_in_blocks = curr_need_swap_in_blocks[:
                                                                budget_bound_num_blocks]

            # 5. try allocate
            allocated_swap_in_blocks = []
            for cpu_block, gpu_block in curr_need_swap_in_blocks:
                assert cpu_block.ready()
                # read from cpu_block
                cpu_block.incr()

                assert gpu_block.ready()
                # write to gpu_block, need acquire lock

                try:
                    self.offloading_manager.gpu_block_allocator.allocate_block(
                        gpu_block)
                except NoFreeBlocksError:
                    break

                gpu_block.acquire()
                allocated_swap_in_blocks.append((cpu_block, gpu_block))

            # 6. add to swap_in_task
            num_blocks = len(allocated_swap_in_blocks)
            swap_in_budget.need_swap_in_blocks.extend(allocated_swap_in_blocks)

            # 7. add to budget
            swap_in_budget.add_num_blocks(request.request_id, num_blocks)
            swap_in_budget.add_num_requests(request.request_id, 1)
