import random
import time

import numpy as np

from wde.microservices.framework.zero.server import ZeroServerProcess
from wde.microservices.standalone.server import setup_and_run
from wde.workflows.decoding.kv_cache.physical_manager import \
    allocate_blockwise_kv_cache
from wde.workflows.decoding.kv_cache.remote.client import \
    ZeroRemoteKVCacheClient
from wde.workflows.decoding.kv_cache.remote.memory import get_share_memory_np


def benchmark_remote_kv_cache_server(name, N, max_num_batched_tokens,
                                     block_size, num_attention_layers,
                                     num_kv_heads, head_size, cache_dtype,
                                     pin_memory):
    server = setup_and_run()
    server_name = "kv_cache_server"

    kv_cache_server = ZeroServerProcess(
        "wde.workflows.decoding.kv_cache.remote.server:ZeroRemoteKVCacheServer",
        server_kwargs={
            "name": server_name,
            "model": name,
            "engine_args": {
                "block_size": 16,
                "memory_space": 10,
                "cache_dtype": "auto"
            }
        })

    kv_cache_server.start()

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

    client = ZeroRemoteKVCacheClient()
    client.wait_service_available(server_name)

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
        block_hash = hash(delta_token_ids)
        block_hashs.append(block_hash)
        hash_map[block_hash] = i

    block_hashs = np.array(block_hashs, dtype=np.int64)

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

    def test_set():

        # warming up
        for i in range(N):
            from_ids = random.sample(block_ids, n)

            from_block_hashs = block_hashs[from_ids]
            blocks = [from_kv_cache_np[f] for f in from_ids]

            client.set(server_name, name, from_block_hashs, blocks, force=True)

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]

            from_block_hashs = block_hashs[from_ids]
            blocks = [from_kv_cache_np[f] for f in from_ids]

            client.set(server_name, name, from_block_hashs, blocks, force=True)

        end = time.perf_counter()
        elapsed_time = end - start

        print("set elapsed time: ", elapsed_time)

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

            count = 0

            for t, rep in zip(to_ids, response):
                data = rep.block
                to_kv_cache_np[t] = data
                count += 1

            assert count == n

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()

        print("stream_get elapsed time: ", elapsed_time)

    try:
        test_set()
        test_contains()
        test_get()
        test_stream_get()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        kv_cache_server.terminate()
        server.terminate()


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    import torch
    max_num_batched_tokens = 1024
    block_size = 16
    N = 8

    for name, num_attention_layers, num_kv_heads, head_size, cache_dtype in [
            # ("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", 64, 8, 128, torch.float16),
        ("Qwen/Qwen2.5-7B-Instruct", 28, 4, 128, torch.bfloat16),
        ("Qwen/Qwen2.5-3B-Instruct", 36, 2, 128, torch.bfloat16),
        ("THUDM/glm-4-9b-chat-1m", 40, 4, 128, torch.bfloat16),
        ("NousResearch/Hermes-3-Llama-3.1-8B", 32, 8, 128, torch.bfloat16),
    ]:
        print(name)
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(
                benchmark_remote_kv_cache_server,
                *(name, N, max_num_batched_tokens, block_size,
                  num_attention_layers, num_kv_heads, head_size, cache_dtype,
                  False))
            f.result()

        time.sleep(10)
"""
Qwen/Qwen2.5-7B-Instruct
set elapsed time:  0.12471285799983889
get elapsed time:  0.16931634599950485
stream_get elapsed time:  0.0733827679996466
"""
