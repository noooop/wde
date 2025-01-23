import random
import time

import torch

from wde.workflows.decoding.kv_cache.physical_manager import (
    allocate_blockwise_kv_cache, allocate_layerwise_kv_cache)


@torch.no_grad()
def benchmark_layerwise_to_blockwise_transfer_blocks(N, max_num_batched_tokens,
                                                     block_size,
                                                     num_attention_layers,
                                                     num_kv_heads, head_size,
                                                     cache_dtype, pin_memory):

    n = max_num_batched_tokens // block_size
    num_blocks = N * n

    gpu_kv_cache = allocate_layerwise_kv_cache(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype,
        pin_memory=False,
        device="cuda")

    cpu_kv_cache = allocate_blockwise_kv_cache(
        num_blocks=num_blocks,
        num_attention_layers=num_attention_layers,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        cache_dtype=cache_dtype,
        pin_memory=pin_memory)

    block_ids = list(range(num_blocks))

    tasks = []
    for i in range(N):
        from_ids = random.sample(block_ids, n)
        to_ids = random.sample(block_ids, n)
        tasks.append((from_ids, to_ids))

    def random_(t):
        if isinstance(t, list):
            for layer in t:
                layer.random_()
        else:
            t.random_()

    def zero_(t):
        if isinstance(t, list):
            for layer in t:
                layer.zero_()
        else:
            t.zero_()

    def test(from_kv_cache, to_kv_cache):
        if isinstance(from_kv_cache, list):
            name = "gpu to cpu"
        else:
            name = "cpu to gpu"

        random_(from_kv_cache)

        def correctness_test():
            from_ids, to_ids = tasks[-1]

            for i in range(num_attention_layers):
                if isinstance(from_kv_cache, list):
                    a = from_kv_cache[i][:, from_ids].cuda()
                    b = to_kv_cache[to_ids, i].permute(1, 0, 2, 3, 4).cuda()
                else:
                    a = from_kv_cache[from_ids, i].permute(1, 0, 2, 3,
                                                           4).cuda()
                    b = to_kv_cache[i][:, to_ids].cuda()

                if not torch.all(torch.isclose(a, b)):
                    assert False

        def test_correctness_test():
            try:
                correctness_test()
            except AssertionError:
                print("correctness_test ok!")

        def test_naive():
            zero_(to_kv_cache)

            start = time.perf_counter()
            if isinstance(from_kv_cache, list):
                for from_ids, to_ids in tasks:
                    for i in range(num_attention_layers):
                        for f, t in zip(from_ids, to_ids):
                            to_kv_cache[t, i, ...] = from_kv_cache[i][:, f,
                                                                      ...]
            else:
                for from_ids, to_ids in tasks:
                    for i in range(num_attention_layers):
                        for f, t in zip(from_ids, to_ids):
                            to_kv_cache[i][:, t] = from_kv_cache[f, i]

            torch.cuda.synchronize()

            end = time.perf_counter()
            elapsed_time = end - start

            correctness_test()
            print(f"{name} b2b naive elapsed time: ", elapsed_time)

        def test_ops():
            from wde.workflows.decoding.kv_cache.offloading.swap import \
                swap_blocks

            zero_(to_kv_cache)

            start = time.perf_counter()

            for from_ids, to_ids in tasks:
                need_swap = list(zip(from_ids, to_ids))
                swap_blocks(from_kv_cache, to_kv_cache, need_swap)

            torch.cuda.synchronize()

            end = time.perf_counter()
            elapsed_time = end - start

            correctness_test()
            print(f"{name} b2b ops.swap_blocks elapsed time: ", elapsed_time)

        test_correctness_test()
        test_naive()
        test_ops()

    test(from_kv_cache=gpu_kv_cache, to_kv_cache=cpu_kv_cache)
    test(from_kv_cache=cpu_kv_cache, to_kv_cache=gpu_kv_cache)


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

        benchmark_layerwise_to_blockwise_transfer_blocks(
            N,
            max_num_batched_tokens,
            block_size,
            num_attention_layers,
            num_kv_heads,
            head_size,
            cache_dtype,
            pin_memory=True)
"""
Qwen/Qwen2.5-7B-Instruct
gpu to cpu b2b naive elapsed time:  0.3089604969991342
gpu to cpu b2b ops.swap_blocks elapsed time:  0.07835619599973143
cpu to gpu b2b naive elapsed time:  0.18914010299886286
cpu to gpu b2b ops.swap_blocks elapsed time:  0.07765093099988007
"""
