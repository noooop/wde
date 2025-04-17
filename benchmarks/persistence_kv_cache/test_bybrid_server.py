import time

import torch
from easydict import EasyDict as edict

from benchmarks.remote_kv_cache.test_server import test
from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             start_remote_kv_cache,
                                             wait_service_available)
from wde.utils import process_warp_with_exc
from wde.workflows.decoding.kv_cache_server.filesystem import rm_cache_dir


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
    args.file_space = 40
    args.kv_cache_folder = "/share/test_kv_cache"

    args.remote_kv_cache_server_name = server_name

    rm_cache_dir(args.model, args.block_size, args.kv_cache_folder)

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

    rm_cache_dir(args.model, args.block_size, args.kv_cache_folder)


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
set deferred=False elapsed time:  0.24872518200027116
set deferred=True elapsed time:  0.08373432000007597
get elapsed time:  0.2531905089999782
stream_get elapsed time:  0.12403806800102757
"""
