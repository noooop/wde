import time
from typing import List

import torch
from vllm import _custom_ops as ops
from vllm.utils import is_pin_memory_available


def allocate_kv_cache(num_blocks, num_attention_layers, block_size,
                      num_kv_heads, head_size, cache_dtype, device):
    pin_memory = is_pin_memory_available() if device == "cpu" else False

    kv_cache_shape = (2, num_blocks, block_size, num_kv_heads, head_size)

    kv_cache: List[torch.Tensor] = []
    for _ in range(num_attention_layers):
        kv_cache.append(
            torch.randn(kv_cache_shape,
                        dtype=cache_dtype,
                        pin_memory=pin_memory,
                        device=device))
    return kv_cache


def swap_blocks(
    src_kv_cache: torch.Tensor,
    dst_kv_cache: torch.Tensor,
    src_to_dst: torch.Tensor,
) -> None:
    src_key_cache = src_kv_cache[0]
    dst_key_cache = dst_kv_cache[0]
    ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
    src_value_cache = src_kv_cache[1]
    dst_value_cache = dst_kv_cache[1]
    ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)


def benchmark_swap_blocks(N, max_num_batched_tokens, block_size,
                          num_attention_layers, num_kv_heads, head_size,
                          cache_dtype):
    n = max_num_batched_tokens // block_size
    num_gpu_blocks = 1 * n
    num_cpu_blocks = N * n

    cpu_kv_cache = allocate_kv_cache(num_blocks=num_cpu_blocks,
                                     num_attention_layers=num_attention_layers,
                                     block_size=block_size,
                                     num_kv_heads=num_kv_heads,
                                     head_size=head_size,
                                     cache_dtype=cache_dtype,
                                     device="cuda")

    gpu_kv_cache = allocate_kv_cache(num_blocks=num_gpu_blocks,
                                     num_attention_layers=num_attention_layers,
                                     block_size=block_size,
                                     num_kv_heads=num_kv_heads,
                                     head_size=head_size,
                                     cache_dtype=cache_dtype,
                                     device="cpu")

    print("swap in")
    start = time.perf_counter()
    for i in range(N):
        block_mapping = [(i + N * j, j) for j in range(n)]
        blocks_to_swap = torch.tensor(block_mapping,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        for l1 in range(num_attention_layers):
            swap_blocks(cpu_kv_cache[l1], gpu_kv_cache[l1], blocks_to_swap)
        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed_time = end - start
    print("elapsed time: ", elapsed_time)

    print("swap out")
    start = time.perf_counter()
    for i in range(N):
        block_mapping = [(j, i + N * j) for j in range(n)]
        blocks_to_swap = torch.tensor(block_mapping,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        for l2 in range(num_attention_layers):
            swap_blocks(gpu_kv_cache[l2], cpu_kv_cache[l2], blocks_to_swap)

        torch.cuda.synchronize()

    end = time.perf_counter()
    elapsed_time = end - start
    print("elapsed time: ", elapsed_time)


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

        benchmark_swap_blocks(N, max_num_batched_tokens, block_size,
                              num_attention_layers, num_kv_heads, head_size,
                              cache_dtype)
