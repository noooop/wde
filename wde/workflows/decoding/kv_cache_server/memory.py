from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.offloading.manager import \
    CPUBlockAllocator
from wde.workflows.decoding.kv_cache.remote.util import (
    GB, MB, allocate_blockwise_kv_cache_np, get_cache_block_size_bytes,
    get_cache_shape)

logger = init_logger(__name__)


class RemoteMemoryKVCache:

    def __init__(self, model, block_size, memory_space, cache_dtype="auto"):
        self.model = model
        self.block_size = block_size
        self.cache_dtype = cache_dtype
        self.memory_space_bytes = memory_space * GB

        num_attention_layers, num_heads, head_size, dtype = get_cache_shape(
            self.model, self.cache_dtype)

        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        self.cache_block_size = get_cache_block_size_bytes(
            num_attention_layers, block_size, num_heads, head_size, dtype)

        self.num_blocks = self.memory_space_bytes // self.cache_block_size

        self.kv_cache = self._allocate_kv_cache()
        self.block_allocator = CPUBlockAllocator(num_blocks=self.num_blocks,
                                                 block_size=self.block_size)
        self.block_shape = self.kv_cache.shape[1:]

        logger.info(
            f"KV cache shape:{self.kv_cache.shape}. KV cache size {self.cache_block_size / MB} MB."
        )

    def __contains__(self, block_hash):
        return block_hash in self.block_allocator

    def __len__(self):
        return len(self.block_allocator)

    @property
    def info(self):
        return self.block_allocator.info

    def get(self, block_hash):
        block = self.block_allocator.get(block_hash)

        return block

    def get_or_create(self, block_hash):
        block = self.block_allocator.create(block_hash)
        return block

    def contains(self, block_hash, refresh):
        o = block_hash in self.block_allocator

        if o and refresh:
            block = self.block_allocator.get(block_hash)
            self.block_allocator.refresh(block)

        return o

    def _allocate_kv_cache(self):
        kv_cache = allocate_blockwise_kv_cache_np(self.num_blocks,
                                                  self.num_attention_layers,
                                                  self.block_size,
                                                  self.num_heads,
                                                  self.head_size, self.dtype)
        return kv_cache
