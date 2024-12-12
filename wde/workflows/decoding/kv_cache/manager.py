from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.physical_manager import \
    PhysicalGPUKVCacheManager
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

logger = init_logger(__name__)

_GB = float(2**30)

RequestId = str


class KVCacheManager:

    def __init__(self, engine_config, physical_kv_cache_manager):
        self.physical_gpu_kv_cache_manager = physical_kv_cache_manager
        self.engine_config = engine_config

    @classmethod
    def from_engine(cls, engine):
        physical_gpu_kv_cache_manager = PhysicalGPUKVCacheManager.from_engine(
            engine)
        return cls(engine_config=engine.engine_config,
                   physical_kv_cache_manager=physical_gpu_kv_cache_manager)

    # logical KVCacheManager api
    def create_vblock(self, request: DecodingSchedulableRequest):
        raise NotImplementedError

    def can_allocate(self, request: DecodingSchedulableRequest,
                     budget_bound_token_chunk_size: int) -> int:
        raise NotImplementedError

    def allocate(self, request: DecodingSchedulableRequest) -> None:
        raise NotImplementedError

    def free(self, request: DecodingSchedulableRequest) -> None:
        raise NotImplementedError

    def free_last_block(self, request: DecodingSchedulableRequest):
        raise NotImplementedError


def raise_if_cache_size_invalid(num_gpu_blocks, block_size,
                                max_model_len) -> None:
    if num_gpu_blocks <= 0:
        raise ValueError("No available memory for the cache blocks. "
                         "Try increasing `gpu_memory_utilization` when "
                         "initializing the engine.")
    max_seq_len = block_size * num_gpu_blocks
    if max_model_len > max_seq_len:
        raise ValueError(
            f"The model's max seq len ({max_model_len}) "
            "is larger than the maximum number of tokens that can be "
            f"stored in KV cache ({max_seq_len}). Try increasing "
            "`gpu_memory_utilization` or decreasing `max_model_len` when "
            "initializing the engine.")
