import numpy as np

from wde.workflows.decoding.kv_cache.remote.memory import dtype_np_map


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
