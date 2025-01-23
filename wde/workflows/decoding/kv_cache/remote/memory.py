import ctypes as C

import numpy as np
import torch

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.offloading.manager import \
    CPUBlockAllocator

_GB = 1 << 30
_MB = 1 << 20

logger = init_logger(__name__)


class RemoteMemoryKVCache:

    def __init__(self, model, block_size, memory_space, cache_dtype="auto"):
        self.model = model
        self.block_size = block_size
        self.cache_dtype = cache_dtype
        self.memory_space_bytes = memory_space * _GB

        self.num_attention_layers, self.num_heads, self.head_size, self.dtype = self._get_cache_shape(
        )

        self.cache_block_size = self._get_cache_block_size_bytes()
        self.num_blocks = self.memory_space_bytes // self.cache_block_size
        self.kv_cache = self._allocate_kv_cache()
        self.block_allocator = CPUBlockAllocator(num_blocks=self.num_blocks,
                                                 block_size=self.block_size)

        logger.info(
            f"KV cache shape:{self.kv_cache.shape}. KV cache size {self.cache_block_size / _MB} MB."
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

    def _get_cache_shape(self):
        num_attention_layers, num_heads, head_size, dtype = process_warp(
            get_kv_cache_shape, model=self.model, cache_dtype=self.cache_dtype)
        return num_attention_layers, num_heads, head_size, dtype

    def _get_cache_block_size_bytes(self) -> int:
        dtype_size = get_dtype_size(self.dtype)

        key_cache_block = self.block_size * self.num_heads * self.head_size
        value_cache_block = key_cache_block
        total = self.num_attention_layers * (key_cache_block +
                                             value_cache_block)
        return dtype_size * total

    def _allocate_kv_cache(self):
        kv_cache = allocate_blockwise_kv_cache_np(self.num_blocks,
                                                  self.num_attention_layers,
                                                  self.block_size,
                                                  self.num_heads,
                                                  self.head_size, self.dtype)
        return kv_cache


# numpy does not support bfloat16
dtype_np_map = {
    torch.float16: np.int16,
    torch.bfloat16: np.int16,
    torch.float32: np.int32,
}

dtype_ctypes_map = {
    torch.float16: C.c_int16,
    torch.bfloat16: C.c_int16,
    torch.float32: C.c_int32,
}


def allocate_blockwise_kv_cache_np(num_blocks, num_attention_layers,
                                   block_size, num_kv_heads, head_size,
                                   cache_dtype):
    assert cache_dtype in dtype_np_map

    kv_cache_shape = (num_blocks, num_attention_layers, 2, block_size,
                      num_kv_heads, head_size)

    kv_cache = np.empty(kv_cache_shape, dtype=dtype_np_map[cache_dtype])

    return kv_cache


def get_share_memory_np(tensor):
    dtype = tensor.dtype

    assert dtype in dtype_ctypes_map

    data_pointer = tensor.data_ptr()
    data_pointer = C.cast(data_pointer, C.POINTER(dtype_ctypes_map[dtype]))
    return np.ctypeslib.as_array(data_pointer, shape=tuple(tensor.shape))


def get_dtype_size(dtype: torch.dtype):
    return torch.tensor([], dtype=dtype).element_size()


def get_kv_cache_shape(model, cache_dtype):
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.WARNING)

    from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

    from wde.workflows.core.config import ModelConfig

    model_config = ModelConfig(model=model,
                               tokenizer=model,
                               tokenizer_mode='auto',
                               trust_remote_code=False,
                               dtype="auto",
                               seed=0)

    head_size = model_config.get_head_size()
    num_attention_layers = model_config.get_num_attention_layers()
    num_kv_heads = model_config.get_num_kv_heads()

    if cache_dtype == "auto":
        dtype = model_config.dtype
    else:
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]

    return num_attention_layers, num_kv_heads, head_size, dtype


def process_warp(fn, /, *args, **kwargs):
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(1) as executor:
        f = executor.submit(fn, *args, **kwargs)
        return f.result()
