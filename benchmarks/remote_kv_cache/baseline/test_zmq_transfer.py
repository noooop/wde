import random
import time
import warnings
from multiprocessing import Process
from threading import Thread

import numpy as np
import torch
import zmq

from wde.workflows.decoding.kv_cache.physical_manager import \
    allocate_blockwise_kv_cache

warnings.filterwarnings("ignore", category=UserWarning)


def server(url, from_kv_cache):
    from_kv_cache = from_kv_cache.numpy()

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.bind(url)

    while True:
        buffer, *_ = socket.recv_multipart(copy=False)
        from_ids = np.frombuffer(buffer.buffer, dtype=np.int32)
        socket.send_multipart([from_kv_cache[f] for f in from_ids], copy=False)


def benchmark_zmq_transfer_blocks(transport, N, max_num_batched_tokens,
                                  block_size, num_attention_layers,
                                  num_kv_heads, head_size, cache_dtype,
                                  pin_memory):

    if transport == "tcp":
        server_url = "tcp://*:5555"
        client_url = "tcp://localhost:5555"
    elif transport == "ipc":
        server_url = "ipc://test_ipc"
        client_url = "ipc://test_ipc"
    elif transport == "inproc":
        server_url = "inproc://test_inproc"
        client_url = "inproc://test_inproc"
    else:
        assert False

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

    if transport == "inproc":
        s = Thread(target=server,
                   args=(
                       server_url,
                       from_kv_cache,
                   ),
                   daemon=True)
    else:
        s = Process(target=server, args=(
            server_url,
            from_kv_cache,
        ))
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

    def test_zmq_transfer():
        buffer_shape = to_kv_cache.shape[1:]
        buffer_dtype = to_kv_cache.dtype

        to_kv_cache.zero_()

        context = zmq.Context.instance()
        socket = context.socket(zmq.REQ)
        socket.connect(client_url)

        # warming up
        for i in range(N):
            from_ids, to_ids = tasks[i]
            socket.send_multipart([from_ids], copy=False)
            socket.recv_multipart(copy=False)

        start = time.perf_counter()

        for i in range(N):
            from_ids, to_ids = tasks[i]
            socket.send_multipart([from_ids], copy=False)
            data = socket.recv_multipart(copy=False)

            for j, t in enumerate(to_ids):
                to_kv_cache[t, ...] = torch.frombuffer(
                    data[j].buffer, dtype=buffer_dtype).view(buffer_shape)

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        print(f"zmq_transfer {transport} elapsed time: ", elapsed_time)

    def test_async_zmq_transfer():
        buffer_shape = to_kv_cache.shape[1:]
        buffer_dtype = to_kv_cache.dtype

        to_kv_cache.zero_()

        context = zmq.Context()
        sockets = []
        for i in range(N):
            socket = context.socket(zmq.REQ)
            socket.connect(client_url)
            sockets.append(socket)

        # warming up
        for i in range(N):
            socket = sockets[i]
            from_ids, to_ids = tasks[i]
            socket.send_multipart([from_ids], copy=False)
            socket.recv_multipart(copy=False)

        start = time.perf_counter()

        for i in range(N):
            socket = sockets[i]
            from_ids, to_ids = tasks[i]
            socket.send_multipart([from_ids], copy=False)

        for i in range(N):
            socket = sockets[i]
            from_ids, to_ids = tasks[i]

            data = socket.recv_multipart(copy=False)

            for j, t in enumerate(to_ids):
                to_kv_cache[t, ...] = torch.frombuffer(
                    data[j].buffer, dtype=buffer_dtype).view(buffer_shape)

        end = time.perf_counter()
        elapsed_time = end - start

        correctness_test()
        print(f"async_zmq_transfer {transport} elapsed time: ", elapsed_time)

    test_correctness_test()
    test_zmq_transfer()

    if not transport == "inproc":
        test_async_zmq_transfer()
        s.terminate()


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    max_num_batched_tokens = 1024
    block_size = 16
    N = 8

    for name, num_attention_layers, num_kv_heads, head_size, cache_dtype in [
            # ("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", 64, 8, 128, torch.float16),
        ("Qwen/Qwen2.5-7B-Instruct", 28, 4, 128, torch.float16),
        ("Qwen/Qwen2.5-3B-Instruct", 36, 2, 128, torch.float16),
        ("THUDM/glm-4-9b-chat-1m-hf", 40, 4, 128, torch.float16),
        ("NousResearch/Hermes-3-Llama-3.1-8B", 32, 8, 128, torch.float16),
    ]:
        print(name)
        for transport in ["tcp", "ipc", "inproc"]:

            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(
                    benchmark_zmq_transfer_blocks,
                    *(transport, N, max_num_batched_tokens, block_size,
                      num_attention_layers, num_kv_heads, head_size,
                      cache_dtype, False))
                f.result()
"""
Qwen/Qwen2.5-7B-Instruct
zmq_transfer tcp elapsed time:  0.1284674380000297
async_zmq_transfer tcp elapsed time:  0.14995783599999868
zmq_transfer ipc elapsed time:  0.11866889700002048
async_zmq_transfer ipc elapsed time:  0.1380156210000223
zmq_transfer inproc elapsed time:  0.039067191000071944
"""
