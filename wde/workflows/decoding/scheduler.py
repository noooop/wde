from wde.workflows.decoding.config import DecodingEngineConfig


def get_scheduler(engine_config: DecodingEngineConfig):
    if engine_config.cache_config.enable_prefix_caching:
        return "wde.workflows.decoding.kv_cache.prefix_caching.scheduler:PrefixCachingDecodingScheduler"
    else:
        return "wde.workflows.decoding.kv_cache.naive.scheduler:NaiveDecodingScheduler"
