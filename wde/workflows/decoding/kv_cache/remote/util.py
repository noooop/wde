import ctypes as C

import numpy as np
import torch

from wde.utils import process_warp

GB = 1 << 30
MB = 1 << 20

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


def allocate_blockwise_kv_cache_mmap(filename, num_blocks,
                                     num_attention_layers, block_size,
                                     num_kv_heads, head_size, cache_dtype):
    assert cache_dtype in dtype_np_map

    kv_cache_shape = (num_blocks, num_attention_layers, 2, block_size,
                      num_kv_heads, head_size)

    kv_cache = np.memmap(filename,
                         dtype=dtype_np_map[cache_dtype],
                         mode='w+',
                         shape=kv_cache_shape)
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
    elif isinstance(cache_dtype, str):
        dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
    else:
        dtype = cache_dtype

    return num_attention_layers, num_kv_heads, head_size, dtype


def get_cache_shape(model, cache_dtype):
    num_attention_layers, num_heads, head_size, dtype = process_warp(
        get_kv_cache_shape, model=model, cache_dtype=cache_dtype)
    return num_attention_layers, num_heads, head_size, dtype


def get_cache_block_size_bytes(num_attention_layers, block_size, num_heads,
                               head_size, dtype):
    dtype_size = get_dtype_size(dtype)

    key_cache_block = block_size * num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    return dtype_size * total
