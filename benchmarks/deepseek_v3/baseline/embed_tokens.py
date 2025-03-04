# ruff: noqa: F841, E402

import time

import torch
from vllm.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding
from vllm.utils import DeviceMemoryProfiler

from wde.utils import process_warp_with_exc

from .util import GB, config, offloading


def init(device="cpu"):
    weights = torch.rand(config.vocab_size,
                         config.hidden_size,
                         dtype=torch.bfloat16,
                         device=device,
                         pin_memory=device == "cpu")

    embed_tokens = VocabParallelEmbedding(
        config.vocab_size,
        config.hidden_size,
    )

    params_dict = dict(embed_tokens.named_parameters())
    embed_tokens.weight_loader(params_dict['weight'], weights)

    return embed_tokens


@torch.inference_mode
def test_cpu(batchsize_list, repeat):
    embed_tokens = init()

    def test(n, repeat):
        requests = []
        for i in range(repeat):
            input_tokens = torch.randint(config.vocab_size, (n, ),
                                         dtype=torch.long,
                                         device="cpu",
                                         pin_memory=True)
            requests.append(input_tokens)

        start = time.perf_counter()
        for i in range(repeat):
            input_tokens = requests[i]
            out = embed_tokens(input_tokens)
        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    for bs in batchsize_list:
        test(bs, repeat=3)
        elapsed_time = test(bs, repeat=repeat)

        print(f"Batchsize: {bs}, "
              f"Throughput: {repeat * bs / elapsed_time:.4f} token/s, "
              f"Latency: {elapsed_time*1000 / repeat:.4f} ms")


@torch.inference_mode
def test_cpu_h2d(batchsize_list, repeat):
    embed_tokens = init()

    def test(n, repeat):
        requests = []
        for i in range(repeat):
            input_tokens = torch.randint(config.vocab_size, (n, ),
                                         dtype=torch.long,
                                         device="cpu",
                                         pin_memory=True)
            requests.append(input_tokens)

        start = time.perf_counter()
        for i in range(repeat):
            input_tokens = requests[i]
            out = embed_tokens(input_tokens)
            out.to("cuda", non_blocking=True)
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    for bs in batchsize_list:
        test(bs, repeat=3)
        elapsed_time = test(bs, repeat=repeat)

        print(f"Batchsize: {bs}, "
              f"Throughput: {repeat * bs / elapsed_time:.4f} token/s, "
              f"Latency: {elapsed_time*1000 / repeat:.4f} ms")


@torch.inference_mode
def test_gpu(batchsize_list, repeat):
    embed_tokens = init(device=config.device)

    def test(n, repeat):
        requests = []
        for i in range(repeat):
            input_tokens = torch.randint(config.vocab_size, (n, ),
                                         dtype=torch.long,
                                         device="cpu",
                                         pin_memory=True)
            requests.append(input_tokens)

        start = time.perf_counter()
        for i in range(repeat):
            input_tokens = requests[i]
            input_tokens.to("cuda", non_blocking=True)
            out = embed_tokens(input_tokens)
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    for bs in batchsize_list:
        test(bs, repeat=3)
        elapsed_time = test(bs, repeat=repeat)

        print(f"Batchsize: {bs}, "
              f"Throughput: {repeat * bs / elapsed_time:.4f} token/s, "
              f"Latency: {elapsed_time*1000 / repeat:.4f} ms")


@torch.inference_mode
def test_offloading(repeat):

    with DeviceMemoryProfiler(config.device) as m:
        with torch.device(config.device):
            model = init()

    model_memory_usage = m.consumed_memory

    def test(repeat):
        start = time.perf_counter()

        for i in range(repeat):
            offloading(model, "cpu", non_blocking=True)
            torch.cuda.synchronize()

            offloading(model, "cuda", non_blocking=True)
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    test(repeat=3)
    elapsed_time = test(repeat=repeat)

    print(
        f"weights_memory: {model_memory_usage / GB:.4f}, "
        f"offloading elapsed time: {elapsed_time*1000 / (repeat * 2):.4f} ms. "
    )


if __name__ == '__main__':
    repeat = 1000
    batchsize_list = [2**i for i in range(12)]

    print("test_cpu:")
    process_warp_with_exc(test_cpu, batchsize_list, repeat)

    print("test_cpu_h2d:")
    process_warp_with_exc(test_cpu_h2d, batchsize_list, repeat)

    print("test_gpu:")
    process_warp_with_exc(test_gpu, batchsize_list, repeat)

    process_warp_with_exc(test_offloading, repeat=10)
