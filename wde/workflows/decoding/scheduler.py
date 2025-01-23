from wde.workflows.decoding.config import DecodingEngineConfig

# BlockAllocators
NAIVE_BLOCK_ALLOCATOR = "wde.workflows.decoding.kv_cache.naive.allocator:NaiveBlockAllocator"
PREFIX_CACHING_BLOCK_ALLOCATOR = "wde.workflows.decoding.kv_cache.prefix_caching.allocator:PrefixCachingBlockAllocator"
DISABLE_PREFIX_CACHING_BLOCK_ALLOCATOR = "wde.workflows.decoding.kv_cache.prefix_caching.allocator:DisablePrefixCachingBlockAllocator"
YOCO_BLOCK_ALLOCATOR = "wde.workflows.decoding.kv_cache.yoco.allocator:YOCOPrefixCachingBlockAllocator"

# Schedulers
NAIVE_SCHEDULER = "wde.workflows.decoding.kv_cache.naive.scheduler:NaiveDecodingScheduler"
PREFIX_CACHING_SCHEDULER = "wde.workflows.decoding.kv_cache.prefix_caching.scheduler:PrefixCachingDecodingScheduler"
OFFLOADING_SCHEDULER = "wde.workflows.decoding.kv_cache.offloading.scheduler:OffloadingKVCachingDecodingScheduler"
REMOTE_KV_CACHR_SCHEDULER = "wde.workflows.decoding.kv_cache.remote.scheduler:RemoteKVCachingDecodingScheduler"

BLOCK_ALLOCATOR_MAP = {
    "naive": NAIVE_BLOCK_ALLOCATOR,
    "prefix_caching": PREFIX_CACHING_BLOCK_ALLOCATOR,
    "disable_prefix_caching": DISABLE_PREFIX_CACHING_BLOCK_ALLOCATOR,
    "yoco": YOCO_BLOCK_ALLOCATOR
}


def get_scheduler_and_block_allocator(engine_config: DecodingEngineConfig):
    if engine_config.cache_config.remote_kv_cache_server_name is not None:
        assert engine_config.cache_config.swap_space_bytes > 0

        if engine_config.cache_config.enable_prefix_caching:
            return REMOTE_KV_CACHR_SCHEDULER, PREFIX_CACHING_BLOCK_ALLOCATOR
        else:
            return REMOTE_KV_CACHR_SCHEDULER, DISABLE_PREFIX_CACHING_BLOCK_ALLOCATOR

    if engine_config.cache_config.block_allocator is not None:
        block_allocator = engine_config.cache_config.block_allocator

        if block_allocator == "naive":
            return NAIVE_SCHEDULER, BLOCK_ALLOCATOR_MAP[block_allocator]

        if block_allocator not in BLOCK_ALLOCATOR_MAP:
            raise ValueError(
                f"block allocator : {block_allocator} not supported")

        if engine_config.cache_config.swap_space_bytes > 0:
            return OFFLOADING_SCHEDULER, BLOCK_ALLOCATOR_MAP[block_allocator]
        else:
            return PREFIX_CACHING_SCHEDULER, BLOCK_ALLOCATOR_MAP[
                block_allocator]

    if engine_config.cache_config.swap_space_bytes > 0:
        if engine_config.cache_config.enable_prefix_caching:
            return OFFLOADING_SCHEDULER, PREFIX_CACHING_BLOCK_ALLOCATOR
        else:
            return OFFLOADING_SCHEDULER, DISABLE_PREFIX_CACHING_BLOCK_ALLOCATOR
    else:
        if engine_config.cache_config.enable_prefix_caching:
            return PREFIX_CACHING_SCHEDULER, PREFIX_CACHING_BLOCK_ALLOCATOR
        else:
            return NAIVE_SCHEDULER, NAIVE_BLOCK_ALLOCATOR
