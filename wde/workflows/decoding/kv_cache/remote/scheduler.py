from typing import Optional

from wde.logger import init_logger
from wde.utils import lazy_import
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.decoding.kv_cache.logic_manager import LogicKVCacheManager
from wde.workflows.decoding.kv_cache.offloading.manager import \
    OffloadingManager
from wde.workflows.decoding.kv_cache.offloading.scheduler import \
    OffloadingKVCachingDecodingScheduler
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
        self.remote_manager.check_finishd_task()

        scheduler_outputs = super().schedule()

        self.remote_manager.add_transfer_callback(
            scheduler_outputs.swap_out_task)

        return scheduler_outputs

    def join(self):
        super().join()
        self.remote_manager.join()
