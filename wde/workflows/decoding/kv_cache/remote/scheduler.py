import time
from typing import Optional

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
    DecodingSwapInSchedulingBudget, OffloadingKVCachingDecodingScheduler)
from wde.workflows.decoding.kv_cache.remote.manager import RemoteManager
from wde.workflows.decoding.schema.engine_io import DecodingSchedulerOutput

logger = init_logger(__name__)


class RemoteKVCachingDecodingScheduler(OffloadingKVCachingDecodingScheduler):
    name = "Remote KV Caching"
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(self, engine_config: EngineConfig,
                 request_processor: RequestProcessor, kv_cache_manager,
                 offloading_manager: OffloadingManager,
                 remote_manager: RemoteManager) -> None:
        super().__init__(engine_config, request_processor, kv_cache_manager,
                         offloading_manager)
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
        self.remote_manager.check_finishd_task()

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

    def join(self):
        self.offloading_manager.join()
        self.remote_manager.join()
