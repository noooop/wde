import random
import time
from multiprocessing import Process

import numpy as np
import torch
import zmq

from wde.workflows.decoding.kv_cache.physical_manager import \
    allocate_blockwise_kv_cache
from wde.workflows.decoding.kv_cache.remote.util import (
    allocate_blockwise_kv_cache_np, get_share_memory_np)


def server(url, num_blocks, block_size, num_attention_layers, num_kv_heads,
           head_size, cache_dtype):

    kv_cache = allocate_blockwise_kv_cache_np(num_blocks, num_attention_layers,
                                              block_size, num_kv_heads,
                                              head_size, cache_dtype)

    buffer_dtype = kv_cache.dtype
    buffer_shape = kv_cache.shape[1:]

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(url)

    def deal_with_send(payload):
        from_ids, *data = payload
        from_ids = np.frombuffer(from_ids.buffer, dtype=np.int32)

        assert len(from_ids) == len(data)

        for i, f in enumerate(from_ids):
            kv_cache[f] = np.frombuffer(
                data[i], dtype=buffer_dtype).reshape(buffer_shape)

        socket.send(b"ok!")

    def deal_with_recv(payload):
        from_ids, *_ = payload
        from_ids = np.frombuffer(from_ids.buffer, dtype=np.int32)
        socket.send_multipart([kv_cache[f] for f in from_ids], copy=False)

    while True:
        method_name, *payload = socket.recv_multipart(copy=False)

        if method_name.bytes == b"send":
            deal_with_send(payload)
        else:
            deal_with_recv(payload)


def benchmark_zmq_transfer_blocks(N, max_num_batched_tokens, block_size,
                                  num_attention_layers, num_kv_heads,
                                  head_size, cache_dtype, pin_memory):
    server_url = "tcp://*:5555"
    client_url = "tcp://localhost:5555"

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
    to_kv_cache.zero_()

    from_kv_cache_np = get_share_memory_np(from_kv_cache)
    to_kv_cache_np = get_share_memory_np(to_kv_cache)

    buffer_shape = to_kv_cache_np.shape[1:]
    buffer_dtype = to_kv_cache_np.dtype

    s = Process(target=server,
                args=(server_url, num_blocks, block_size, num_attention_layers,
                      num_kv_heads, head_size, cache_dtype))
    s.start()

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

    def test_correctness_test():
        try:
            correctness_test()
        except AssertionError:
            print("correctness_test ok!")

    def test_send():
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.connect(client_url)

        # warming up
        for i in range(N):
            from_ids, to_ids = tasks[i]
            msg_parts = [b"send", from_ids
                         ] + [from_kv_cache_np[f] for f in from_ids]
            socket.send_multipart(msg_parts, copy=False)
            socket.recv_multipart(copy=False)

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]
            socket.send_multipart([b"send", from_ids] +
                                  [from_kv_cache_np[f] for f in from_ids],
                                  copy=False)
            socket.recv_multipart(copy=False)

        end = time.perf_counter()
        elapsed_time = end - start

        print("zmq send elapsed time: ", elapsed_time)

    def test_recv():
        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.connect(client_url)

        # warming up
        for i in range(N):
            from_ids, to_ids = tasks[i]
            socket.send_multipart([b"recv", from_ids], copy=False)
            socket.recv_multipart(copy=False)

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]
            socket.send_multipart([b"recv", from_ids], copy=False)

            data = socket.recv_multipart(copy=False)

            assert len(from_ids) == len(data)

            for j, t in enumerate(to_ids):
                to_kv_cache_np[t] = np.frombuffer(
                    data[j].buffer, dtype=buffer_dtype).reshape(buffer_shape)

        end = time.perf_counter()
        elapsed_time = end - start

        print("zmq recv elapsed time: ", elapsed_time)

    test_correctness_test()

    test_send()
    test_recv()

    try:
        correctness_test()
    finally:
        s.terminate()


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
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
                benchmark_zmq_transfer_blocks,
                *(N, max_num_batched_tokens, block_size, num_attention_layers,
                  num_kv_heads, head_size, cache_dtype, False))
            f.result()
"""
Qwen/Qwen2.5-7B-Instruct
zmq send elapsed time:  0.09121608699933859
zmq recv elapsed time:  0.17693168400001014
"""
