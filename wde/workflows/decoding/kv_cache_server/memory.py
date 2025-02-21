from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.offloading.manager import \
    CPUBlockAllocator
from wde.workflows.decoding.kv_cache.prefix_caching.util import \
    block_hashs_to_numpy_array
from wde.workflows.decoding.kv_cache.remote.util import (
    GB, MB, allocate_blockwise_kv_cache_np, get_cache_block_size_bytes,
    get_cache_shape)
from wde.workflows.decoding.kv_cache_server.Interface import \
    RemoteKVCacheInterface

logger = init_logger(__name__)


class RemoteMemoryKVCache(RemoteKVCacheInterface):

    def __init__(self,
                 model,
                 block_size,
                 memory_space,
                 cache_dtype="auto",
                 *args,
                 **kwargs):
        self.model = model
        self.block_size = block_size
        self.cache_dtype = cache_dtype
        self.memory_space_bytes = int(memory_space * GB)

        self.num_attention_layers = None
        self.num_heads = None
        self.head_size = None
        self.dtype = None
        self.cache_block_size = None
        self.num_blocks = None
        self.kv_cache = None
        self.block_allocator = None
        self.block_shape = None

    def init(self):
        num_attention_layers, num_heads, head_size, dtype = get_cache_shape(
            self.model, self.cache_dtype)

        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype

        self.cache_block_size = get_cache_block_size_bytes(
            num_attention_layers, self.block_size, num_heads, head_size, dtype)

        self.num_blocks = self.memory_space_bytes // self.cache_block_size

        self.kv_cache = self._allocate_kv_cache()
        self.block_allocator = CPUBlockAllocator(num_blocks=self.num_blocks,
                                                 block_size=self.block_size)
        self.block_shape = self.kv_cache.shape[1:]

        logger.info(
            f"KV cache shape:{self.kv_cache.shape}. KV cache size {self.cache_block_size / MB} MB."
        )

    def set(self, block_hashs, block_data, force):
        block_shape = self.block_shape
        block_allocator = self.block_allocator

        total = len(block_hashs)
        blocks = {}

        error = 0
        existed = 0
        forced = 0
        created = 0
        duplicated = 0

        for i in range(total):
            block_hash = block_hashs[i].tobytes()

            if block_hash in blocks:
                duplicated += 1
                continue

            data = block_data[i]

            assert data.shape == block_shape

            block = block_allocator.get_or_create(block_hash)

            if block is None:
                error += 0
                # NoFreeBlocksError
                continue

            if block.lock:
                existed += 1
                # doing write
                continue

            if block.lock is None:
                created += 1
            else:
                existed += 1

                if not force:
                    continue
                else:
                    forced += 1

            block.acquire()

            block_allocator.hold(block)
            blocks[block_hash] = (block, data)

        def generator():
            for block, data in blocks.values():
                self.kv_cache[block.physical_block_id] = data

        def release():
            for block, data in blocks.values():
                block.release()
                block_allocator.free(block)

        info = {
            "total": total,
            "error": error,
            "existed": existed,
            "duplicated": duplicated,
            "created": created,
            "forced": forced
        }

        return info, generator, release

    def contains(self, block_hashs, refresh):
        total = len(block_hashs)
        block_allocator = self.block_allocator

        hit = []
        miss = []

        for i in range(total):
            block_hash = block_hashs[i].tobytes()

            h = block_hash in block_allocator

            if h and refresh:
                block = block_allocator.get(block_hash)
                block_allocator.refresh(block)

            if h:
                hit.append(block_hash)
            else:
                miss.append(block_hash)

        hit = block_hashs_to_numpy_array(hit)
        miss = block_hashs_to_numpy_array(miss)

        return hit, miss

    def get(self, block_hashs):
        block_allocator = self.block_allocator

        total = len(block_hashs)
        hit = 0
        miss = 0
        duplicate = 0

        blocks = {}
        for i in range(total):
            block_hash = block_hashs[i].tobytes()

            if block_hash in blocks:
                duplicate += 1
                continue

            block = block_allocator.get(block_hash)

            if block is None:
                miss += 1
                continue

            hit += 1
            block_allocator.hold(block)

            blocks[block_hash] = block

        def generator():
            for block_hash, block in blocks.items():
                data = self.kv_cache[block.physical_block_id]
                yield block_hash, data

        def release():
            for block in blocks.values():
                block_allocator.free(block)

        info = {
            "total": total,
            "hit": hit,
            "miss": miss,
            "duplicate": duplicate,
        }

        return info, generator, release

    def __contains__(self, block_hash):
        return block_hash in self.block_allocator

    def __len__(self):
        return len(self.block_allocator)

    @property
    def info(self):
        return self.block_allocator.info

    def _allocate_kv_cache(self):
        kv_cache = allocate_blockwise_kv_cache_np(self.num_blocks,
                                                  self.num_attention_layers,
                                                  self.block_size,
                                                  self.num_heads,
                                                  self.head_size, self.dtype)
        return kv_cache
