from wde.workflows.decoding.config import DecodingEngineConfig

NAIVE = "wde.workflows.decoding.kv_cache.naive.scheduler:NaiveDecodingScheduler"
PREFIX_CACHING = "wde.workflows.decoding.kv_cache.prefix_caching.scheduler:PrefixCachingDecodingScheduler"
YOCO = "wde.workflows.decoding.kv_cache.yoco.scheduler:YOCOPrefixCachingDecodingScheduler"
OFFLOADING_KV_CACHING = "wde.workflows.decoding.kv_cache.offloading.scheduler:OffloadingKVCachingDecodingScheduler"

KV_CACHE_MANAGER_MAP = {
    "naive": NAIVE,
    "prefix_caching": PREFIX_CACHING,
    "yoco": YOCO
}


def get_scheduler(engine_config: DecodingEngineConfig):
    if engine_config.cache_config.swap_space_bytes > 0:
        return OFFLOADING_KV_CACHING

    if engine_config.cache_config.kv_cache_manager is not None:
        kv_cache_manager = engine_config.cache_config.kv_cache_manager
        if kv_cache_manager not in KV_CACHE_MANAGER_MAP:
            raise ValueError(
                f"kv cache manager : {kv_cache_manager} not supported")

        return KV_CACHE_MANAGER_MAP[kv_cache_manager]
    elif engine_config.cache_config.enable_prefix_caching:
        return PREFIX_CACHING
    else:
        return NAIVE
