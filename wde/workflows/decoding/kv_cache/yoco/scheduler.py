from wde.workflows.decoding.kv_cache.prefix_caching.scheduler import \
    PrefixCachingDecodingScheduler
from wde.workflows.decoding.kv_cache.yoco.manager import \
    YOCOPrefixCachingBlockAllocator


class YOCOPrefixCachingDecodingScheduler(PrefixCachingDecodingScheduler):
    # You only compute once Prefix Caching

    name = "YOCO Prefix Caching"
    support_scheduling = ["sync_scheduling", "async_scheduling"]
    block_allocator_class = YOCOPrefixCachingBlockAllocator
