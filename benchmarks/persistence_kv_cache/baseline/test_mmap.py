import random
import time

import numpy as np
import torch

from benchmarks.remote_kv_cache.util import process_warp_with_exc
from wde.workflows.decoding.kv_cache.persistence.mmap import \
    allocate_blockwise_kv_cache_mmap
from wde.workflows.decoding.kv_cache.remote.memory import \
    allocate_blockwise_kv_cache_np


def benchmark_mmap_transfer_blocks(N, max_num_batched_tokens, block_size,
                                   num_attention_layers, num_kv_heads,
                                   head_size, cache_dtype, pin_memory):
    filename = "test_mmap.block"

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

    memory_kv_cache = allocate_blockwise_kv_cache_np(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype)

    persistence_kv_cache = allocate_blockwise_kv_cache_mmap(
        filename, num_blocks, num_attention_layers, block_size, num_kv_heads,
        head_size, cache_dtype)

    def set_random(t):
        t[:] = np.random.randn(*t.shape)
        if hasattr(t, "flush"):
            t.flush()

    def set_zero(t):
        t[:] = np.zeros_like(t)
        if hasattr(t, "flush"):
            t.flush()

    def benchmark(from_kv_cache, to_kv_cache):
        # numpy dont have inplace operate random_ zero_
        set_random(from_kv_cache)

        block_ids = list(range(num_blocks))

        tasks = []
        for i in range(N):
            from_ids = random.sample(block_ids, n)
            to_ids = random.sample(block_ids, n)
            tasks.append((from_ids, to_ids))

        def correctness_test():
            from_ids, to_ids = tasks[-1]
            if not np.all(
                    np.isclose(from_kv_cache[from_ids], to_kv_cache[to_ids])):
                assert False

        set_zero(to_kv_cache)

        start = time.perf_counter()

        for from_ids, to_ids in tasks:
            for f, t in zip(from_ids, to_ids):
                to_kv_cache[t, ...] = from_kv_cache[f, ...]

        if hasattr(to_kv_cache, "flush"):
            to_kv_cache.flush()

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        return elapsed_time

    def test_memory_to_persistence():
        elapsed_time = benchmark(from_kv_cache=memory_kv_cache,
                                 to_kv_cache=persistence_kv_cache)

        print("memory_to_persistence elapsed time: ", elapsed_time)

    def test_persistence_to_memory():
        elapsed_time = benchmark(from_kv_cache=memory_kv_cache,
                                 to_kv_cache=persistence_kv_cache)

        print("persistence_to_memory elapsed time: ", elapsed_time)

    test_memory_to_persistence()
    test_persistence_to_memory()


if __name__ == '__main__':
    max_num_batched_tokens = 1024
    block_size = 16
    N = 8

    for name, num_attention_layers, num_kv_heads, head_size, cache_dtype in [
        ("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", 64, 8, 128, torch.float16),
        ("Qwen/Qwen2.5-7B-Instruct", 28, 4, 128, torch.bfloat16),
        ("Qwen/Qwen2.5-3B-Instruct", 36, 2, 128, torch.bfloat16),
        ("THUDM/glm-4-9b-chat-1m", 40, 4, 128, torch.bfloat16),
        ("NousResearch/Hermes-3-Llama-3.1-8B", 32, 8, 128, torch.bfloat16),
    ]:
        print(name)
        process_warp_with_exc(
            benchmark_mmap_transfer_blocks,
            *(N, max_num_batched_tokens, block_size, num_attention_layers,
              num_kv_heads, head_size, cache_dtype, False))
"""
Qwen/Qwen2.5-7B-Instruct
memory_to_persistence elapsed time:  0.13666347499929543
persistence_to_memory elapsed time:  0.14086130900068383
"""
