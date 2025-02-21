import time

import pytest
from easydict import EasyDict as edict

from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             start_remote_kv_cache,
                                             wait_service_available)
from wde.utils import process_warp
from wde.workflows.decoding.kv_cache.physical_manager import \
    allocate_blockwise_kv_cache
from wde.workflows.decoding.kv_cache.prefix_caching.util import (
    block_hashs_to_numpy_array, get_block_hash, get_prefix_hash)
from wde.workflows.decoding.kv_cache.remote.util import (get_cache_shape,
                                                         get_share_memory_np)
from wde.workflows.decoding.kv_cache_server.client import \
    ZeroRemoteKVCacheClient
from wde.workflows.decoding.kv_cache_server.filesystem import (cache_dir_files,
                                                               rm_cache_dir)

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


def main(args):
    process_warp(wait_service_available, args)

    init_prefix_str = f"kv_cache:{args.model}:{args.block_size}"
    init_prefix_hash = get_prefix_hash(init_prefix_str.encode())

    info = process_warp(kv_cache_info, args)

    assert info.block_size == args.block_size
    assert info.num_full_blocks == 0
    assert info.num_blocks == info.num_free_blocks

    num_blocks = info.num_blocks

    num_attention_layers, num_heads, head_size, dtype = get_cache_shape(
        args.model, args.cache_dtype)

    kv_cache = allocate_blockwise_kv_cache(
        num_blocks=1,
        num_attention_layers=num_attention_layers,
        block_size=args.block_size,
        num_kv_heads=num_heads,
        head_size=head_size,
        cache_dtype=dtype,
        pin_memory=False)

    kv_cache.random_()

    blocks = [get_share_memory_np(kv_cache)[0]]

    client = ZeroRemoteKVCacheClient()

    # not full
    for i in range(1, num_blocks):
        block_hash = get_block_hash(init_prefix_hash, [i])
        block_hashs = block_hashs_to_numpy_array([block_hash])

        response = client.set(args.server_name,
                              args.model,
                              block_hashs,
                              blocks,
                              force=True,
                              deferred=False)

        assert response.error == 0
        assert response.duplicated == 0
        assert response.total == 1
        assert response.existed + response.created == 1

        if i % 100 == 0:
            time.sleep(0.01)

            info = process_warp(kv_cache_info, args)

            assert info.num_full_blocks == i
            assert len(
                cache_dir_files(args.model, args.block_size,
                                args.kv_cache_folder)) == i

    # full
    for i in range(1, num_blocks):
        block_hash = get_block_hash(init_prefix_hash, [i + num_blocks])
        block_hashs = block_hashs_to_numpy_array([block_hash])

        response = client.set(args.server_name,
                              args.model,
                              block_hashs,
                              blocks,
                              force=True,
                              deferred=False)

        assert response.error == 0
        assert response.duplicated == 0
        assert response.total == 1
        assert response.existed + response.created == 1

        if i % 100 == 0:
            time.sleep(0.01)

            info = process_warp(kv_cache_info, args)

            assert info.num_free_blocks == 0
            assert info.num_full_blocks == num_blocks
            assert len(
                cache_dir_files(args.model, args.block_size,
                                args.kv_cache_folder)) == num_blocks


@pytest.mark.parametrize("model", MODELS)
def test_evict(model: str):
    server_name = "remote_kv_cache_server"

    args = edict()
    args.model = model
    args.server_name = server_name
    args.remote_kv_cache_server_name = server_name
    args.block_size = 16
    args.cache_dtype = "auto"
    args.memory_space = 0
    args.file_space = 0.1
    args.kv_cache_folder = "/share/test_kv_cache"

    rm_cache_dir(model, args.block_size, args.kv_cache_folder)
    assert len(cache_dir_files(model, args.block_size,
                               args.kv_cache_folder)) == 0

    server = start_remote_kv_cache(args)

    try:
        process_warp(main, args)
    finally:
        for s in server:
            s.terminate()

        rm_cache_dir(model, args.block_size, args.kv_cache_folder)

    time.sleep(10)
