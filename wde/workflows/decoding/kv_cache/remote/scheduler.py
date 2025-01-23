import time
from collections import deque
from typing import List, Optional

from wde.logger import init_logger
from wde.utils import lazy_import
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.decoding.kv_cache.logic_manager import LogicKVCacheManager
from wde.workflows.decoding.kv_cache.naive.scheduler import \
    DecodingSchedulingBudget
from wde.workflows.decoding.kv_cache.offloading.manager import \
    OffloadingManager
from wde.workflows.decoding.kv_cache.offloading.scheduler import (
    DecodingSwapInSchedulingBudget, SchedulerSwapInRunningOutputs,
    SchedulerSwapInWaitingOutputs)
from wde.workflows.decoding.kv_cache.prefix_caching.scheduler import \
    PrefixCachingDecodingScheduler
from wde.workflows.decoding.kv_cache.remote.manager import RemoteManager
from wde.workflows.decoding.schema.engine_io import DecodingSchedulerOutput
from wde.workflows.decoding.schema.request import (DecodingSchedulableRequest,
                                                   RequestStatus)

logger = init_logger(__name__)


class RemoteKVCachingDecodingScheduler(PrefixCachingDecodingScheduler):
    name = "Remote KV Caching"
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(self, engine_config: EngineConfig,
                 request_processor: RequestProcessor, kv_cache_manager,
                 offloading_manager: OffloadingManager,
                 remote_manager: RemoteManager) -> None:
        super().__init__(engine_config, request_processor, kv_cache_manager)
        self.offloading_manager = offloading_manager
        self.remote_manager = remote_manager

    @classmethod
    def from_engine(cls, engine):
        block_allocator_class = lazy_import(engine.workflow.BlockAllocator)
        engine_config = engine.engine_config

        gpu_kv_cache_manager = LogicKVCacheManager.from_engine(
            engine=engine, block_allocator_class=block_allocator_class)

        cpu_cache = engine.kv_cache_manager.cpu_cache
        gpu_cache = engine.kv_cache_manager.gpu_cache

        offloading_manager = OffloadingManager(
            engine_config=engine_config,
            cpu_cache=cpu_cache,
            gpu_cache=gpu_cache,
            gpu_block_allocator=gpu_kv_cache_manager.block_allocator)

        remote_manager = RemoteManager(
            engine_config=engine_config,
            cpu_cache=cpu_cache,
            cpu_block_allocator=offloading_manager.cpu_block_allocator)

        return cls(engine_config,
                   engine.request_processor,
                   kv_cache_manager=gpu_kv_cache_manager,
                   offloading_manager=offloading_manager,
                   remote_manager=remote_manager)

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

        swap_out_task = self.offloading_manager.get_swap_out_task(
            scheduler_outputs)
        self.remote_manager.add_transfer_callback(swap_out_task)
        scheduler_outputs.swap_out_task = swap_out_task

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

        budget = DecodingSchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        # schedule running
        swap_in_runnings = self._schedule_swap_in_runnings(swap_in_budget)
        running_queue = swap_in_runnings.running_queue
        running_scheduled = self._schedule_running(budget, running_queue)

        # schedule waiting
        waiting_scheduled = self._schedule_waiting(swap_in_budget, budget)

        self.running = running_queue
        self.running.extend(running_scheduled.decode_requests)
        self.running.extend(running_scheduled.prefill_requests)
        self.running.extend(running_scheduled.preempted)
        self.running.extend(swap_in_runnings.busy_requests)
        self.running.extend(waiting_scheduled.scheduled_requests)
        self.running.extend(waiting_scheduled.all_swap_in_able_swap_in_ed)

        self.waiting.extendleft(
            waiting_scheduled.not_all_swap_in_able_swap_in_ed)

        scheduled_requests = []
        scheduled_requests.extend(running_scheduled.decode_requests)
        scheduled_requests.extend(running_scheduled.prefill_requests)
        scheduled_requests.extend(waiting_scheduled.scheduled_requests)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_requests <= self.scheduler_config.max_num_requests

        return DecodingSchedulerOutput(
            scheduled_requests=scheduled_requests,
            num_batched_tokens=budget.num_batched_tokens,
            num_requests=budget.num_curr_requests,
            ignored_requests=waiting_scheduled.ignored_requests,
            need_swap_in_blocks=swap_in_budget.need_swap_in_blocks)

    def _schedule_waiting(self, swap_in_budget,
                          budget) -> SchedulerSwapInWaitingOutputs:
        # for normal prefill
        ignored_requests: List[DecodingSchedulableRequest] = []
        scheduled_requests: List[DecodingSchedulableRequest] = []

        # for swap_in
        all_swap_in_able_swap_in_ed: List[DecodingSchedulableRequest] = []
        not_all_swap_in_able_swap_in_ed: List[DecodingSchedulableRequest] = []

        if not self.waiting:
            return SchedulerSwapInWaitingOutputs(
                scheduled_requests=scheduled_requests,
                ignored_requests=ignored_requests,
                all_swap_in_able_swap_in_ed=all_swap_in_able_swap_in_ed,
                not_all_swap_in_able_swap_in_ed=not_all_swap_in_able_swap_in_ed
            )

        waiting_queue = self.waiting
        while waiting_queue:
            if budget.full():
                break

            if swap_in_budget.full():
                break

            if self.kv_cache_manager.high_watermark():
                break

            request = waiting_queue[0]

            if self.record_metrics:
                scheduled_ts = time.perf_counter()

            # 1. Check if it has been aborted
            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                waiting_queue.popleft()
                continue

            # 2. Check if request been preprocessored
            if not isinstance(request, DecodingSchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            # The busy flag in waiting_queue now acts as whether it is preprocessed
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

                # 4. create vblock
                self.kv_cache_manager.create(request)

                # 5. try to hit prefix caching
                self.kv_cache_manager.update(request)

                request.busy = True

            # 6. find blocks need swap in
            curr_need_swap_in_blocks = self.offloading_manager.get_swap_in_blocks(
                request)

            if not curr_need_swap_in_blocks:
                # all_swap_in_able_swap_in_ed
                # normal prefill

                # N1. Check if hit prefix caching ready
                if not request.vblock.ready():
                    waiting_queue.popleft()

                    # move to running_queue
                    request.busy = False
                    all_swap_in_able_swap_in_ed.append(request)

                    if len(all_swap_in_able_swap_in_ed
                           ) > budget.max_num_requests:
                        # too many not ready requests.
                        # no need to continue exploring
                        # maybe all requests have the same prefix
                        # We don't need to confirm that all requests are not ready
                        break
                    continue

                # N2. chunked prefill
                num_new_tokens = request.num_new_tokens
                budget_bound_token_chunk_size = min(
                    num_new_tokens, budget.remaining_token_budget())

                if budget_bound_token_chunk_size == 0:
                    # No budget => Stop
                    break

                # N3. try to allocate gpu blocks
                memory_bound_token_chunk_size = self.kv_cache_manager.can_allocate(
                    request, budget_bound_token_chunk_size)

                if memory_bound_token_chunk_size == 0:
                    # No free gpu blocks => Stop
                    break

                # N4. Allocate kv cache
                request.token_chunk_size = memory_bound_token_chunk_size
                self.kv_cache_manager.allocate(request)

                # N5. lock all block avoid recompute
                request.vblock.acquire()

                # N6. Can schedule this request.
                waiting_queue.popleft()

                # N7. set running
                request.status = RequestStatus.RUNNING

                scheduled_requests.append(request)
                budget.add_num_batched_tokens(request.request_id,
                                              request.token_chunk_size)
                budget.add_num_requests(request.request_id, 1)

                if self.record_metrics:
                    request.set_scheduled_ts(scheduled_ts)

            else:
                # need swap_in
                waiting_queue.popleft()

                # S1. meet budget
                num_need_swap_in_blocks = len(curr_need_swap_in_blocks)

                budget_bound_num_blocks = min(
                    num_need_swap_in_blocks,
                    swap_in_budget.remaining_blocks_budget())

                curr_need_swap_in_blocks = curr_need_swap_in_blocks[:
                                                                    budget_bound_num_blocks]

                # S2. try to allocate & lock blocks avoid recompute
                allocated_swap_in_blocks = self.offloading_manager.try_allocate_swap_in_blocks(
                    curr_need_swap_in_blocks)

                # S3. add to swap_in_task
                num_blocks = len(allocated_swap_in_blocks)
                swap_in_budget.need_swap_in_blocks.extend(
                    allocated_swap_in_blocks)

                # S4. add to budget
                swap_in_budget.add_num_blocks(request.request_id, num_blocks)
                swap_in_budget.add_num_requests(request.request_id, 1)

                if num_need_swap_in_blocks == num_blocks:
                    # move to running_queue
                    request.busy = False
                    all_swap_in_able_swap_in_ed.append(request)
                else:
                    # The busy flag in waiting_queue now acts as whether it is preprocessed
                    # move to waiting_queue
                    not_all_swap_in_able_swap_in_ed.append(request)

        return SchedulerSwapInWaitingOutputs(
            scheduled_requests=scheduled_requests,
            ignored_requests=ignored_requests,
            all_swap_in_able_swap_in_ed=all_swap_in_able_swap_in_ed,
            not_all_swap_in_able_swap_in_ed=not_all_swap_in_able_swap_in_ed)

    def _schedule_swap_in_runnings(self, swap_in_budget):
        busy_requests = []
        running_queue = []

        if not self.running:
            return SchedulerSwapInRunningOutputs(
                busy_requests=busy_requests,
                running_queue=deque(running_queue))

        for request in self.running:
            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                continue

            if request.request_id not in self.requests:
                # aborted_requests
                continue

            if request.busy:
                busy_requests.append(request)
            else:
                running_queue.append(request)

        running = deque(
            sorted(running_queue, key=lambda request: request.arrival_time))

        running_queue = deque()

        while running:
            if swap_in_budget.full():
                break

            if self.kv_cache_manager.high_watermark():
                break

            request = running.popleft()

            # 1. Write new token ids to vblock & try to hit prefix caching
            self.kv_cache_manager.update(request)

            # 2. find blocks need swap in
            curr_need_swap_in_blocks = self.offloading_manager.get_swap_in_blocks(
                request)

            # 3. no need swap in
            if not curr_need_swap_in_blocks:
                if not request.vblock.ready():
                    busy_requests.append(request)
                else:
                    running_queue.append(request)
                continue

            # 4. meet budget
            num_need_swap_in_blocks = len(curr_need_swap_in_blocks)

            budget_bound_num_blocks = min(
                num_need_swap_in_blocks,
                swap_in_budget.remaining_blocks_budget())

            curr_need_swap_in_blocks = curr_need_swap_in_blocks[:
                                                                budget_bound_num_blocks]

            # 5. try to allocate & lock blocks avoid recompute
            allocated_swap_in_blocks = self.offloading_manager.try_allocate_swap_in_blocks(
                curr_need_swap_in_blocks)

            # 6. add to swap_in_task
            num_blocks = len(allocated_swap_in_blocks)
            swap_in_budget.need_swap_in_blocks.extend(allocated_swap_in_blocks)

            # 7. add to budget
            swap_in_budget.add_num_blocks(request.request_id, num_blocks)
            swap_in_budget.add_num_requests(request.request_id, 1)

            # 8. add to busy_requests
            busy_requests.append(request)

        running_queue.extend(running)

        return SchedulerSwapInRunningOutputs(
            busy_requests=busy_requests, running_queue=deque(running_queue))

    def join(self):
        self.offloading_manager.join()
        self.remote_manager.join()
