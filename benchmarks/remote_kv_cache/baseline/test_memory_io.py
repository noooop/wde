import random
import time

import torch

from wde.workflows.decoding.kv_cache.physical_manager import (
    allocate_blockwise_kv_cache, allocate_layerwise_kv_cache)


@torch.no_grad()
def benchmark_layerwise_to_layerwise_transfer_blocks(N, max_num_batched_tokens,
                                                     block_size,
                                                     num_attention_layers,
                                                     num_kv_heads, head_size,
                                                     cache_dtype, pin_memory):

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

    from_kv_cache = allocate_layerwise_kv_cache(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype,
        device="cpu",
        pin_memory=pin_memory)

    to_kv_cache = allocate_layerwise_kv_cache(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype,
        device="cpu",
        pin_memory=pin_memory)

    block_ids = list(range(num_blocks))

    tasks = []
    for i in range(N):
        from_ids = random.sample(block_ids, n)
        to_ids = random.sample(block_ids, n)
        tasks.append((from_ids, to_ids))

    def correctness_test():
        for from_ids, to_ids in tasks:
            for layer in range(num_attention_layers):
                for f, t in zip(from_ids, to_ids):
                    if torch.all((to_kv_cache[layer][:, t, ...] -
                                  from_kv_cache[layer][:, f, ...]) > 0.001):
                        assert False

    def test_naive():
        start = time.perf_counter()
        for from_ids, to_ids in tasks:
            for layer in range(num_attention_layers):
                for f, t in zip(from_ids, to_ids):
                    to_kv_cache[layer][:, t, ...] = from_kv_cache[layer][:, f,
                                                                         ...]

        end = time.perf_counter()
        elapsed_time = end - start
        correctness_test()

        print("l2l naive elapsed time: ", elapsed_time)

    test_naive()


@torch.no_grad()
def benchmark_blockwise_to_blockwise_transfer_blocks(N, max_num_batched_tokens,
                                                     block_size,
                                                     num_attention_layers,
                                                     num_kv_heads, head_size,
                                                     cache_dtype, pin_memory):

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

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

    block_ids = list(range(num_blocks))

    tasks = []
    for i in range(N):
        from_ids = random.sample(block_ids, n)
        to_ids = random.sample(block_ids, n)
        tasks.append((from_ids, to_ids))

    def correctness_test():
        from_ids, to_ids = tasks[-1]
        if not torch.all(
                torch.isclose(from_kv_cache[from_ids], to_kv_cache[to_ids])):
            assert False

    def test_correctness_test():
        try:
            correctness_test()
        except AssertionError:
            print("correctness_test ok!")

    def test_naive():
        to_kv_cache.zero_()

        start = time.perf_counter()

        for from_ids, to_ids in tasks:
            for f, t in zip(from_ids, to_ids):
                to_kv_cache[t, ...] = from_kv_cache[f, ...]

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        print("b2b naive elapsed time: ", elapsed_time)

    def test_fancy_index():
        to_kv_cache.zero_()

        start = time.perf_counter()

        for from_ids, to_ids in tasks:
            to_kv_cache[to_ids, ...] = from_kv_cache[from_ids, ...]

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        print("b2b fancy index elapsed time: ", elapsed_time)

    def test_index_copy_():
        to_kv_cache.zero_()

        start = time.perf_counter()

        for from_ids, to_ids in tasks:
            to_kv_cache.index_copy_(dim=0,
                                    index=torch.tensor(to_ids,
                                                       dtype=torch.int64),
                                    source=from_kv_cache[from_ids, ...])

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        print("b2b index_copy_ elapsed time: ", elapsed_time)

    def test_cython_memcpy():
        import numpy as np
        import pyximport

        pyximport.install(setup_args={"include_dirs": np.get_include()})

        from benchmarks.remote_kv_cache.baseline.memcpy import cython_memcpy
        from wde.workflows.decoding.kv_cache.remote.memory import \
            get_share_memory_np

        to_kv_cache.zero_()

        start = time.perf_counter()

        tasks_np = np.array(tasks, dtype=np.int32)
        from_kv_cache_np = get_share_memory_np(from_kv_cache)
        to_kv_cache_np = get_share_memory_np(to_kv_cache)

        cython_memcpy(from_kv_cache_np, to_kv_cache_np, tasks_np)

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        print("b2b cython memcpy elapsed time: ", elapsed_time)

    test_correctness_test()
    test_naive()
    test_fancy_index()
    test_index_copy_()
    test_cython_memcpy()


if __name__ == '__main__':
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

        benchmark_layerwise_to_layerwise_transfer_blocks(
            N,
            max_num_batched_tokens,
            block_size,
            num_attention_layers,
            num_kv_heads,
            head_size,
            cache_dtype,
            pin_memory=False)

        benchmark_blockwise_to_blockwise_transfer_blocks(
            N,
            max_num_batched_tokens,
            block_size,
            num_attention_layers,
            num_kv_heads,
            head_size,
            cache_dtype,
            pin_memory=False)
"""
Qwen/Qwen2.5-7B-Instruct
l2l naive elapsed time:  0.08620247100043343
b2b naive elapsed time:  0.029708318000302825
b2b fancy index elapsed time:  0.07098750899967854
b2b index_copy_ elapsed time:  0.07137100100044336
b2b cython memcpy elapsed time:  0.035842751000018325
"""
