import hashlib
import pathlib

import numpy as np

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.offloading.manager import \
    CPUBlockAllocator
from wde.workflows.decoding.kv_cache.remote.util import GB, MB, dtype_np_map
from wde.workflows.decoding.kv_cache_server.memory import (
    get_cache_block_size_bytes, get_cache_shape)

logger = init_logger(__name__)


class RemoteFilesystemKVCache:

    def __init__(self,
                 model,
                 block_size,
                 file_space,
                 file_dir,
                 cache_dtype="auto",
                 *args,
                 **kwargs):
        self.model = model
        self.block_size = block_size
        self.cache_dtype = cache_dtype
        self.file_space_bytes = file_space * GB

        self._salt_str = f"kvcache:{model}:{block_size}"
        self._salt = hashlib.md5(self._salt_str.encode("UTF-8")).hexdigest()
        self._salt_bytes = self._salt.encode("UTF-8")
        self._filename = self._salt

        self.file_dir = pathlib.Path(file_dir) / self._filename

        self.num_attention_layers = None
        self.num_heads = None
        self.head_size = None
        self.dtype = None
        self.cache_block_size = None
        self.num_blocks = None
        self.block_shape = None
        self.block_allocator = None

    def init(self):
        self.file_dir.mkdir(parents=True, exist_ok=True)

        if not self.file_dir.exists():
            raise FileNotFoundError("kv cache file dir Unable to write")

        num_attention_layers, num_heads, head_size, dtype = get_cache_shape(
            self.model, self.cache_dtype)

        self.num_attention_layers = num_attention_layers
        self.num_heads = num_heads
        self.head_size = head_size
        self.dtype = dtype_np_map[dtype]
        self.block_shape = (num_attention_layers, 2, self.block_size,
                            num_heads, head_size)

        self.cache_block_size = get_cache_block_size_bytes(
            num_attention_layers, self.block_size, num_heads, head_size, dtype)

        self.num_blocks = self.file_space_bytes // self.cache_block_size

        self.block_allocator = CPUBlockAllocator(num_blocks=self.num_blocks,
                                                 block_size=self.block_size)

        self._save_info()
        self._recover()

        logger.info(
            f"KV cache block shape: {self.block_shape}, dtype: {dtype}. KV cache block size {self.cache_block_size / MB} MB."
        )
        logger.info(f"Max num file blocks {self.num_blocks}")

    def get(self, block_hashs):
        block_allocator = self.block_allocator
        block_shape = self.block_shape
        dtype = self.dtype

        total = len(block_hashs)
        hit = 0
        miss = 0
        duplicate = 0

        blocks = {}
        for i in range(total):
            block_hash = block_hashs[i]

            if block_hash in blocks:
                duplicate += 1
                continue

            s_block_hash = self._get_s_block_hash(block_hashs, i)

            block = block_allocator.get(s_block_hash)

            if block is None:
                miss += 1
                continue

            hit += 1
            block_allocator.hold(block)

            blocks[block_hash] = (
                block,
                i,
                s_block_hash,
            )

        def generator():
            for block, index, s_block_hash in blocks.values():
                block_hash = block_hashs[index:index + 1]

                directory = (self.file_dir / s_block_hash[:2] /
                             s_block_hash[2:4])

                data = np.load(directory / (s_block_hash + ".npy"))

                assert data.shape == block_shape
                assert data.dtype == dtype

                yield block_hash, data

        def release():
            for block, index, s_block_hash in blocks.values():
                block_allocator.free(block)

        info = {
            "total": total,
            "hit": hit,
            "miss": miss,
            "duplicate": duplicate,
        }

        return info, blocks, generator, release

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
            block_hash = block_hashs[i]

            if block_hash in blocks:
                duplicated += 1
                continue

            data = block_data[i]

            assert data.shape == block_shape

            s_block_hash = self._get_s_block_hash(block_hashs, i)

            block = block_allocator.get_or_create(s_block_hash)

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
            blocks[block_hash] = (block, data, s_block_hash)

        def generator():
            for block, data, s_block_hash in blocks.values():
                directory = (self.file_dir / s_block_hash[:2] /
                             s_block_hash[2:4])
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / s_block_hash, data)

        def release():
            for block, data, s_block_hash in blocks.values():
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

        return info, blocks, generator, release

    def contains(self, block_hashs, refresh):
        total = len(block_hashs)
        dtype = block_hashs.dtype

        block_allocator = self.block_allocator

        hit = []
        miss = []

        for i in range(total):
            block_hash = block_hashs[i]

            s_block_hash = self._get_s_block_hash(block_hashs, i)

            h = s_block_hash in block_allocator

            if h and refresh:
                block = block_allocator.get(s_block_hash)
                block_allocator.refresh(block)

            if h:
                hit.append(block_hash)
            else:
                miss.append(block_hash)

        hit = np.array(hit, dtype=dtype)
        miss = np.array(miss, dtype=dtype)

        return hit, miss

    @property
    def info(self):
        return self.block_allocator.info

    def __contains__(self, block_hash):
        return block_hash in self.block_allocator

    def __len__(self):
        return len(self.block_allocator)

    def _get_s_block_hash(self, block_hashs, i):
        return hashlib.md5(self._salt_bytes +
                           block_hashs[i:i + 1].tobytes()).hexdigest()

    def _recover(self):

        blocks = []
        for f in self.file_dir.glob("*/*/*"):
            if not f.is_file():
                continue

            if f.suffix != ".npy":
                continue

            stat = f.stat()

            # npy is larger than cache_block_size
            if not self.cache_block_size < stat.st_size < self.cache_block_size * 1.01:
                continue

            s_block_hash = f.stem
            st_atime = stat.st_atime

            blocks.append((st_atime, s_block_hash))

        blocks.sort()

        block_allocator = self.block_allocator
        for s_block_hash, st_atime in blocks:
            block = block_allocator.get_or_create(s_block_hash)
            block.acquire()
            block_allocator.hold(block)
            block.release()
            block_allocator.free(block)

        logger.info("recover %d blocks. info: %s", len(blocks), self.info)

    def _save_info(self):
        with open(self.file_dir / "README.md", "w") as f:
            f.write(f"""
## Warning

This directory is for storing kv cache files. 

Please do not modify any files.

## Info
- model name: {self.model}
- KV cache block shape: {self.block_shape}, dtype: {self.dtype}
- KV cache block size {self.cache_block_size / MB} MB
- Max num file blocks {self.num_blocks}

""")
