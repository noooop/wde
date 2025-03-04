# ruff: noqa: F841, E402

import gc
import time

import torch
from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLP
from vllm.utils import DeviceMemoryProfiler, MemorySnapshot, memory_profiling

from benchmarks.deepseek_v3.baseline.util import (GB, config, get_mlp_weights,
                                                  load_weights, offloading,
                                                  quant_config)


def init():
    weights = get_mlp_weights()

    prefix = "model.layers.0.mlp."

    with torch.device(config.device):
        model = DeepseekV2MLP(hidden_size=config.hidden_size,
                              intermediate_size=config.intermediate_size,
                              hidden_act=config.hidden_act,
                              quant_config=quant_config,
                              prefix=prefix)
        load_weights(model, weights, prefix=prefix)

    return model


@torch.inference_mode
def test_mlp(bs, repeat):
    with DeviceMemoryProfiler(config.device) as m:
        model = init()

    model_memory_usage = m.consumed_memory

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline_snapshot = MemorySnapshot()

    def test(bs, repeat):
        inputs = []
        for i in range(repeat):
            x = torch.rand(bs,
                           config.hidden_size,
                           dtype=torch.bfloat16,
                           device="cuda")
            inputs.append(x)

        start = time.perf_counter()
        for i in range(repeat):
            x = inputs[i]
            out = model(x)
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    with memory_profiling(baseline_snapshot,
                          weights_memory=model_memory_usage) as result:

        test(bs, repeat=3)
        elapsed_time = test(bs, repeat=repeat)

    print(f"Batchsize: {bs}, "
          f"Throughput: {repeat * bs / elapsed_time:.4f} token/s, "
          f"Latency: {elapsed_time*1000 / repeat:.4f} ms. "
          f"weights_memory: {result.weights_memory / GB:.4f}, "
          f"torch_peak_increase: {result.torch_peak_increase / GB:.4f}")


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
    from wde.utils import process_warp_with_exc

    repeat = 100

    batchsize_list = [2**i for i in range(14)]

    for bs in batchsize_list:
        process_warp_with_exc(test_mlp, bs, repeat)

    process_warp_with_exc(test_offloading, repeat=10)
