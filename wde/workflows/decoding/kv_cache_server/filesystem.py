import hashlib
import pathlib

import numpy as np

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.offloading.manager import \
    CPUBlockAllocator
from wde.workflows.decoding.kv_cache.prefix_caching.util import \
    block_hashs_to_numpy_array
from wde.workflows.decoding.kv_cache.remote.util import GB, MB, dtype_np_map
from wde.workflows.decoding.kv_cache_server.Interface import \
    RemoteKVCacheInterface
from wde.workflows.decoding.kv_cache_server.memory import (
    get_cache_block_size_bytes, get_cache_shape)

logger = init_logger(__name__)


class RemoteFilesystemKVCache(RemoteKVCacheInterface):

    def __init__(self,
                 model,
                 block_size,
                 file_space,
                 kv_cache_folder,
                 cache_dtype="auto",
                 *args,
                 **kwargs):
        self.model = model
        self.block_size = block_size
        self.cache_dtype = cache_dtype
        self.file_space_bytes = int(file_space * GB)

        self._salt_str = f"kvcache:{model}:{block_size}"
        self._salt = hashlib.md5(self._salt_str.encode("UTF-8")).hexdigest()
        self._salt_bytes = self._salt.encode("UTF-8")
        self._folder_name = self._salt

        self.kv_cache_folder = pathlib.Path(
            kv_cache_folder) / self._folder_name

        self.num_attention_layers = None
        self.num_heads = None
        self.head_size = None
        self.dtype = None
        self.cache_block_size = None
        self.num_blocks = None
        self.block_shape = None
        self.block_allocator = None

    def init(self):
        self.kv_cache_folder.mkdir(parents=True, exist_ok=True)

        if not self.kv_cache_folder.exists():
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

        self.block_allocator.add_evict_block_callback(
            self.evict_block_callback)
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
                block_hash_str = block_hash.decode("UTF-8")
                directory = (self.kv_cache_folder / block_hash_str[:2] /
                             block_hash_str[2:4])

                data = np.load(directory / (block_hash_str + ".npy"))

                assert data.shape == block_shape
                assert data.dtype == dtype

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
            for block_hash, (block, data) in blocks.items():
                block_hash_str = block_hash.decode("UTF-8")
                directory = (self.kv_cache_folder / block_hash_str[:2] /
                             block_hash_str[2:4])
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / block_hash_str, data)

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

    @property
    def info(self):
        return self.block_allocator.info

    def __contains__(self, block_hash):
        return block_hash in self.block_allocator

    def __len__(self):
        return len(self.block_allocator)

    def evict_block_callback(self, block):
        block_hash = block.block_hash
        block_hash_str = block_hash.decode("UTF-8")
        directory = (self.kv_cache_folder / block_hash_str[:2] /
                     block_hash_str[2:4])
        filepath = directory / (block_hash_str + ".npy")
        filepath.unlink()

    def _recover(self):

        blocks = []
        for f in self.kv_cache_folder.glob("*/*/*"):
            if not f.is_file():
                continue

            if f.suffix != ".npy":
                continue

            stat = f.stat()

            # npy is larger than cache_block_size
            if not self.cache_block_size < stat.st_size < self.cache_block_size * 1.01:
                continue

            block_hash = f.stem.encode("UTF-8")
            st_atime = stat.st_atime

            blocks.append((st_atime, block_hash))

        blocks.sort()

        block_allocator = self.block_allocator
        for st_atime, block_hash in blocks:
            block = block_allocator.get_or_create(block_hash)
            block.lock = False
            block_allocator.refresh(block)

        logger.info("recover %d blocks. info: %s", len(blocks), self.info)

    def _save_info(self):
        with open(self.kv_cache_folder / "README.md", "w") as f:
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


def rm_cache_dir(model, block_size, kv_cache_folder):
    import shutil

    salt_str = f"kvcache:{model}:{block_size}"
    folder_name = hashlib.md5(salt_str.encode("UTF-8")).hexdigest()
    folder_path = pathlib.Path(kv_cache_folder) / folder_name

    try:
        shutil.rmtree(str(folder_path))
    except FileNotFoundError:
        pass


def cache_dir_files(model, block_size, kv_cache_folder):
    salt_str = f"kvcache:{model}:{block_size}"
    folder_name = hashlib.md5(salt_str.encode("UTF-8")).hexdigest()
    folder_path = pathlib.Path(kv_cache_folder) / folder_name

    files = []
    for f in folder_path.glob("*/*/*"):
        if not f.is_file():
            continue

        if f.suffix != ".npy":
            continue

        files.append(f)

    return files
