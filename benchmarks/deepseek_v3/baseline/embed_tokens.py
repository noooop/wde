# ruff: noqa: F841

import time

import torch

from wde.workflows.core.backends.vocab_embedding import VocabParallelEmbedding

vocab_size = 129280
hidden_size = 7168


def test_cpu(batchsize_list, repeat):
    weights = torch.rand(vocab_size,
                         hidden_size,
                         dtype=torch.bfloat16,
                         device="cpu")

    embed_tokens = VocabParallelEmbedding(
        vocab_size,
        hidden_size,
    )

    params_dict = dict(embed_tokens.named_parameters())

    embed_tokens.weight_loader(params_dict['weight'], weights)

    def test(n, repeat):
        requests = []
        for i in range(repeat):
            input_tokens = torch.randint(vocab_size, (n, ),
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

    for batchsize in batchsize_list:

        test(batchsize, repeat=3)
        elapsed_time = test(batchsize, repeat=repeat)

        print(
            f"Batchsize: {batchsize}, Throughput: {repeat * batchsize / elapsed_time:.4f} token/s"
        )


def test_cpu_h2d(batchsize_list, repeat):
    weights = torch.rand(vocab_size,
                         hidden_size,
                         dtype=torch.bfloat16,
                         device="cpu")

    embed_tokens = VocabParallelEmbedding(
        vocab_size,
        hidden_size,
    )

    params_dict = dict(embed_tokens.named_parameters())

    embed_tokens.weight_loader(params_dict['weight'], weights)

    def test(n, repeat):
        requests = []
        for i in range(repeat):
            input_tokens = torch.randint(vocab_size, (n, ),
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

    for batchsize in batchsize_list:

        test(batchsize, repeat=3)
        elapsed_time = test(batchsize, repeat=repeat)

        print(
            f"Batchsize: {batchsize}, Throughput: {repeat * batchsize / elapsed_time:.4f} token/s"
        )


def test_gpu(batchsize_list, repeat):
    weights = torch.rand(vocab_size,
                         hidden_size,
                         dtype=torch.bfloat16,
                         device="cuda")

    embed_tokens = VocabParallelEmbedding(
        vocab_size,
        hidden_size,
    )

    params_dict = dict(embed_tokens.named_parameters())

    embed_tokens.weight_loader(params_dict['weight'], weights)

    def test(n, repeat):
        requests = []
        for i in range(repeat):
            input_tokens = torch.randint(vocab_size, (n, ),
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

    for batchsize in batchsize_list:
        test(batchsize, repeat=3)
        elapsed_time = test(batchsize, repeat=repeat)

        print(
            f"Batchsize: {batchsize}, Throughput: {repeat * batchsize / elapsed_time:.4f} token/s"
        )


if __name__ == '__main__':
    repeat = 1000
    batchsize_list = [2**i for i in range(12)]

    print("test_cpu:")
    test_cpu(batchsize_list, repeat)

    print("test_cpu_h2d:")
    test_cpu_h2d(batchsize_list, repeat)

    print("test_gpu:")
    test_gpu(batchsize_list, repeat)
