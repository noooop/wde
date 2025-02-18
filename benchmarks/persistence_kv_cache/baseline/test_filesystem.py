import hashlib
import random
import time
from pathlib import Path

import numpy as np
import torch

from benchmarks.remote_kv_cache.util import process_warp_with_exc
from wde.workflows.decoding.kv_cache.remote.memory import \
    allocate_blockwise_kv_cache_np


def benchmark_filesystem(name, N, max_num_batched_tokens, block_size,
                         num_attention_layers, num_kv_heads, head_size,
                         cache_dtype, pin_memory):

    cache_dir = Path("cache") / name.replace("/", "__")
    cache_dir.mkdir(exist_ok=True, parents=True)

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

    memory_kv_cache = allocate_blockwise_kv_cache_np(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype)

    def test_memory_to_persistence():
        block_ids = list(range(num_blocks))

        tasks = []
        for i in range(N):
            from_ids = random.sample(block_ids, n)
            to_ids = random.sample(block_ids, n)
            tasks.append((from_ids, to_ids))

        start = time.perf_counter()

        for from_ids, to_ids in tasks:
            for f, t in zip(from_ids, to_ids):
                data = memory_kv_cache[f]
                filename = f"kvcache:{name}:{block_size}:{t}"
                filename = hashlib.md5(filename.encode("UTF-8")).hexdigest()
                np.save(cache_dir / filename, data)

        end = time.perf_counter()
        elapsed_time = end - start

        print("memory_to_persistence elapsed time: ", elapsed_time)

    def test_persistence_to_memory():
        data = memory_kv_cache[0]
        tmp = np.zeros_like(data)

        for i in range(num_blocks):
            filename = f"kvcache:{name}:{block_size}:{i}"
            filename = hashlib.md5(filename.encode("UTF-8")).hexdigest()
            tmp[:] = np.random.randn(*tmp.shape)
            np.save(cache_dir / filename, tmp)

        block_ids = list(range(num_blocks))

        tasks = []
        for i in range(N):
            from_ids = random.sample(block_ids, n)
            to_ids = random.sample(block_ids, n)
            tasks.append((from_ids, to_ids))

        start = time.perf_counter()

        for from_ids, to_ids in tasks:
            for f, t in zip(from_ids, to_ids):
                filename = f"kvcache:{name}:{block_size}:{f}"
                filename = hashlib.md5(filename.encode("UTF-8")).hexdigest()
                memory_kv_cache[t] = np.load(cache_dir / (filename + ".npy"))

        end = time.perf_counter()
        elapsed_time = end - start

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
            benchmark_filesystem,
            *(name, N, max_num_batched_tokens, block_size,
              num_attention_layers, num_kv_heads, head_size, cache_dtype,
              False))
