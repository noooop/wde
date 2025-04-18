import random
import time

import numpy as np
import torch
from easydict import EasyDict as edict

from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             start_remote_kv_cache,
                                             wait_service_available)
from wde.utils import process_warp_with_exc
from wde.workflows.decoding.kv_cache.physical_manager import \
    allocate_blockwise_kv_cache
from wde.workflows.decoding.kv_cache.prefix_caching.util import (
    block_hashs_to_numpy_array, get_block_hash, get_prefix_hash)
from wde.workflows.decoding.kv_cache.remote.util import get_share_memory_np
from wde.workflows.decoding.kv_cache_server.client import \
    ZeroRemoteKVCacheClient


def test(server_name, name, N, n, num_blocks, block_size, num_attention_layers,
         num_kv_heads, head_size, cache_dtype, pin_memory):

    init_prefix_str = f"kv_cache:{name}:{block_size}"
    init_prefix_hash = get_prefix_hash(init_prefix_str.encode())

    client = ZeroRemoteKVCacheClient()

    from_kv_cache = allocate_blockwise_kv_cache(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype,
        pin_memory=pin_memory)

    to_kv_cache = allocate_blockwise_kv_cache(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype,
        pin_memory=pin_memory)

    from_kv_cache.random_()

    from_kv_cache_np = get_share_memory_np(from_kv_cache)
    to_kv_cache_np = get_share_memory_np(to_kv_cache)

    hash_map = {}
    block_hashs = []
    for i in range(num_blocks):
        delta_token_ids = tuple(
            random.randint(1, 1000000) for i in range(block_size))
        block_hash = get_block_hash(init_prefix_hash, delta_token_ids)
        block_hashs.append(block_hash)
        hash_map[block_hash] = i

    block_hashs = block_hashs_to_numpy_array(block_hashs)

    block_ids = list(range(num_blocks))

    tasks = []
    for i in range(N):
        from_ids = random.sample(block_ids, n)
        to_ids = random.sample(block_ids, n)
        tasks.append((from_ids, to_ids))

    tasks = np.array(tasks, dtype=np.int32)

    def correctness_test():
        from_ids, to_ids = tasks[-1]
        if not torch.all(
                torch.isclose(from_kv_cache[from_ids], to_kv_cache[to_ids])):
            assert False

    def test_set(deferred):

        # warming up
        for i in range(N):
            from_ids = random.sample(block_ids, n)

            from_block_hashs = block_hashs[from_ids]
            blocks = [from_kv_cache_np[f] for f in from_ids]

            response = client.set(server_name,
                                  name,
                                  from_block_hashs,
                                  blocks,
                                  force=True,
                                  deferred=deferred)

            assert response.error == 0
            assert response.duplicated == 0
            assert response.total == n
            # assert response.existed == response.forced
            assert response.existed + response.created == n

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]

            from_block_hashs = block_hashs[from_ids]
            blocks = [from_kv_cache_np[f] for f in from_ids]

            response = client.set(server_name,
                                  name,
                                  from_block_hashs,
                                  blocks,
                                  force=True,
                                  deferred=deferred)

            assert response.error == 0
            assert response.duplicated == 0
            assert response.total == n
            # assert response.existed == response.forced
            assert response.existed + response.created == n

        end = time.perf_counter()
        elapsed_time = end - start

        print(f"set deferred={deferred} elapsed time: ", elapsed_time)

    def test_contains():
        for i in range(N):
            from_ids, to_ids = tasks[i]
            from_block_hashs = block_hashs[from_ids]

            response = client.contains(server_name, name, from_block_hashs)

            assert len(response.hit) == n
            assert len(response.miss) == 0

    def test_get():
        # warming up
        for i in range(N):
            from_ids, to_ids = tasks[i]
            from_block_hashs = block_hashs[from_ids]

            client.get(server_name, name, from_block_hashs)

        to_kv_cache.zero_()

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]
            from_block_hashs = block_hashs[from_ids]

            response = client.get(server_name, name, from_block_hashs)
            data = response.blocks

            assert len(response.block_hashs) == n
            assert len(data) == n

            for j, t in enumerate(to_ids):
                to_kv_cache_np[t] = data[j]

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()

        print("get elapsed time: ", elapsed_time)

    def test_stream_get():
        to_kv_cache.zero_()

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]
            from_block_hashs = block_hashs[from_ids]

            response = client.get(server_name,
                                  name,
                                  from_block_hashs,
                                  stream=True)
            metadata = next(response)

            count = 0

            for t, rep in zip(to_ids, response):
                data = rep.block
                to_kv_cache_np[t] = data
                count += 1

            assert count == n
            assert metadata.hit == n

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()

        print("stream_get elapsed time: ", elapsed_time)

    test_set(deferred=False)
    test_set(deferred=True)
    test_contains()
    test_get()
    test_stream_get()


def benchmark_remote_kv_cache_server(name, N, max_num_batched_tokens,
                                     block_size, num_attention_layers,
                                     num_kv_heads, head_size, cache_dtype,
                                     pin_memory):
    server_name = "remote_kv_cache_server"

    args = edict()
    args.model = name
    args.server_name = server_name
    args.block_size = block_size
    args.cache_dtype = cache_dtype
    args.memory_space = 40
    args.remote_kv_cache_server_name = server_name

    server = start_remote_kv_cache(args)
    process_warp_with_exc(wait_service_available, args)

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

    process_warp_with_exc(kv_cache_info, args)
    process_warp_with_exc(test, server_name, name, N, n, num_blocks,
                          block_size, num_attention_layers, num_kv_heads,
                          head_size, cache_dtype, pin_memory)
    process_warp_with_exc(kv_cache_info, args)

    for s in server:
        s.terminate()


if __name__ == '__main__':
    max_num_batched_tokens = 1024
    block_size = 16
    N = 8

    for name, num_attention_layers, num_kv_heads, head_size, cache_dtype in [
        ("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", 64, 8, 128, torch.float16),
        ("Qwen/Qwen2.5-7B-Instruct", 28, 4, 128, torch.bfloat16),
        ("Qwen/Qwen2.5-3B-Instruct", 36, 2, 128, torch.bfloat16),
        ("THUDM/glm-4-9b-chat-1m-hf", 40, 4, 128, torch.bfloat16),
        ("NousResearch/Hermes-3-Llama-3.1-8B", 32, 8, 128, torch.bfloat16),
    ]:
        print(name)
        process_warp_with_exc(
            benchmark_remote_kv_cache_server,
            *(name, N, max_num_batched_tokens, block_size,
              num_attention_layers, num_kv_heads, head_size, cache_dtype,
              False))
        time.sleep(10)
"""
Qwen/Qwen2.5-7B-Instruct
set deferred=False elapsed time:  0.12506789100007154
set deferred=True elapsed time:  0.08860314100093092
get elapsed time:  0.20406127300157095
stream_get elapsed time:  0.06670699400092417
"""
