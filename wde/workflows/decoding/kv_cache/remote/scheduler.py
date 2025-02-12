from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from wde.logger import init_logger
from wde.utils import lazy_import
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.decoding.kv_cache.logic_manager import LogicKVCacheManager
from wde.workflows.decoding.kv_cache.naive.scheduler import \
    SchedulerWaitingOutputs
from wde.workflows.decoding.kv_cache.offloading.manager import \
    OffloadingManager
from wde.workflows.decoding.kv_cache.offloading.scheduler import (
    DecodingSchedulerOutputWithSwapOutTask,
    OffloadingKVCachingDecodingScheduler)
from wde.workflows.decoding.kv_cache.remote.manager import RemoteManager

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.remote.transfer import TransferInTask

logger = init_logger(__name__)


@dataclass
class SchedulerWaitingOutputsWithTransferInTask(SchedulerWaitingOutputs):
    transfer_in_task: Optional["TransferInTask"] = None


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

    def schedule(self) -> Optional[DecodingSchedulerOutputWithSwapOutTask]:
        self.remote_manager.check_finishd_task()

        scheduler_outputs = super().schedule()

        swap_out_task = getattr(scheduler_outputs, "swap_out_task", None)
        self.remote_manager.add_transfer_callback(swap_out_task)

        transfer_in_task = getattr(scheduler_outputs.waiting_scheduled,
                                   "transfer_in_task", None)
        if transfer_in_task is not None and scheduler_outputs.is_empty():
            transfer_in_task.wait()

        return scheduler_outputs

    def _schedule_waiting(self):
        waiting_scheduled = super()._schedule_waiting()

        if not waiting_scheduled.scheduled_requests:
            return waiting_scheduled

        need_transfer_in_blocks = {}
        need_transfer_in_requests = []

        for request in waiting_scheduled.scheduled_requests:
            self.kv_cache_manager.update(request)
            blocks = self.remote_manager.get_transfer_in_blocks(request)

            if len(blocks) == 0:
                continue

            need_transfer_in_requests.append(request)

            for block in blocks:
                if block.block_hash not in need_transfer_in_blocks:
                    need_transfer_in_blocks[block.block_hash] = block

        transfer_in_task = self.remote_manager.get_transfer_in_task(
            need_transfer_in_blocks, need_transfer_in_requests)

        waiting_scheduled = SchedulerWaitingOutputsWithTransferInTask(
            scheduled_requests=waiting_scheduled.scheduled_requests,
            ignored_requests=waiting_scheduled.ignored_requests,
            transfer_in_task=transfer_in_task)

        return waiting_scheduled

    def join(self):
        super().join()
        self.remote_manager.join()
