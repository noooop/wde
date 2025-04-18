# ruff: noqa: F841, E402

import os

os.environ["VLLM_USE_V1"] = "0"

import gc
import os
import time

import torch
from vllm import forward_context
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.utils import DeviceMemoryProfiler, MemorySnapshot, memory_profiling

from benchmarks.deepseek_v3.baseline.util import (GB, cache_config, config,
                                                  get_backbone_layer_weights,
                                                  hf_config, load_weights,
                                                  model_config, offloading,
                                                  quant_config)
from wde.tasks.decode_only.modelzoo.deepseek_v2 import DeepseekV2DecoderLayer


def init():
    os.environ["VLLM_MLA_DISABLE"] = "1"

    kv_cache = torch.rand(2,
                          config.n_blocks,
                          config.block_size,
                          config.num_attention_heads,
                          config.qk_head_dim,
                          dtype=config.dtype,
                          device=config.device)

    weights = get_backbone_layer_weights(wo_moe=True)

    prefix = "model.layers.3"

    with DeviceMemoryProfiler(config.device) as m:
        with torch.device(config.device):
            model = DeepseekV2DecoderLayer(config=hf_config,
                                           model_config=model_config,
                                           cache_config=cache_config,
                                           quant_config=quant_config,
                                           prefix=prefix,
                                           remote_moe=True)

            load_weights(model, weights, prefix=prefix + ".")

            attn = model.self_attn.attn

            attn.use_output = True
            attn.kv_cache = [kv_cache]
            attn.use_direct_call = False

    model_memory_usage = m.consumed_memory

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline_snapshot = MemorySnapshot()

    return model, kv_cache, model_memory_usage, baseline_snapshot


@torch.inference_mode
def test_prefill(n, repeat):
    seq_lens = [n]
    start_loc = [0, n]

    attn_metadata = FlashAttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=n,
        num_decode_tokens=0,
        slot_mapping=torch.tensor(range(n),
                                  dtype=torch.long,
                                  device=config.device),
        seq_lens=seq_lens,
        seq_lens_tensor=torch.tensor(seq_lens,
                                     dtype=torch.int,
                                     device=config.device),
        max_prefill_seq_len=n,
        max_decode_seq_len=0,
        context_lens_tensor=torch.tensor([0],
                                         dtype=torch.int,
                                         device=config.device),
        block_tables=torch.tensor([], device=config.device, dtype=torch.int32),
        max_query_len=n,
        max_decode_query_len=1,
        query_start_loc=torch.tensor(start_loc,
                                     dtype=torch.int32,
                                     device=config.device),
        seq_start_loc=torch.tensor(start_loc,
                                   dtype=torch.int32,
                                   device=config.device),
        use_cuda_graph=False,
        multi_modal_placeholder_index_maps={},
        enable_kv_scales_calculation=False)

    model, kv_cache, model_memory_usage, baseline_snapshot = init()

    forward_context._forward_context = forward_context.ForwardContext(
        no_compile_layers={
            'model.layers.3.self_attn.attn': model.self_attn.attn
        },
        attn_metadata=attn_metadata,
        virtual_engine=0)

    positions = torch.tensor(range(n), dtype=torch.long, device=config.device)
    hidden_states = torch.rand(n,
                               config.hidden_size,
                               dtype=torch.bfloat16,
                               device=config.device)
    residual = torch.rand(n,
                          config.hidden_size,
                          dtype=torch.bfloat16,
                          device=config.device)

    inputs = {
        "positions": positions,
        "hidden_states": hidden_states,
        "residual": residual
    }

    def test(repeat):
        start = time.perf_counter()

        for i in range(repeat):
            out = model(**inputs)
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    with memory_profiling(baseline_snapshot,
                          weights_memory=model_memory_usage) as result:

        test(repeat=3)
        elapsed_time = test(repeat=repeat)

    print(f"Batchsize: {n}, "
          f"Throughput: {repeat * n / elapsed_time:.4f} token/s, "
          f"Latency: {elapsed_time*1000 / repeat:.4f} ms. "
          f"weights_memory: {result.weights_memory / GB:.4f}, "
          f"torch_peak_increase: {result.torch_peak_increase / GB:.4f}")


@torch.inference_mode
def test_decoding(n, repeat):

    seq_lens = [n + 1]
    query_start_loc = [0, 1]
    seq_start_loc = [0, n + 1]

    attn_metadata = FlashAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=torch.tensor([n + 1],
                                  dtype=torch.long,
                                  device=config.device),
        seq_lens=seq_lens,
        seq_lens_tensor=torch.tensor(seq_lens,
                                     dtype=torch.int,
                                     device=config.device),
        max_prefill_seq_len=0,
        max_decode_seq_len=n + 1,
        context_lens_tensor=torch.tensor([n],
                                         dtype=torch.int,
                                         device=config.device),
        block_tables=torch.tensor([range(n)],
                                  device=config.device,
                                  dtype=torch.int32),
        max_query_len=1,
        max_decode_query_len=1,
        query_start_loc=torch.tensor(query_start_loc,
                                     dtype=torch.int32,
                                     device=config.device),
        seq_start_loc=torch.tensor(seq_start_loc,
                                   dtype=torch.int32,
                                   device=config.device),
        use_cuda_graph=False,
        multi_modal_placeholder_index_maps={},
        enable_kv_scales_calculation=False)

    model, kv_cache, model_memory_usage, baseline_snapshot = init()

    forward_context._forward_context = forward_context.ForwardContext(
        no_compile_layers={
            'model.layers.3.self_attn.attn': model.self_attn.attn
        },
        attn_metadata=attn_metadata,
        virtual_engine=0)

    positions = torch.tensor(range(1), dtype=torch.long, device=config.device)
    hidden_states = torch.rand(1,
                               config.hidden_size,
                               dtype=torch.bfloat16,
                               device=config.device)
    residual = torch.rand(1,
                          config.hidden_size,
                          dtype=torch.bfloat16,
                          device=config.device)

    inputs = {
        "positions": positions,
        "hidden_states": hidden_states,
        "residual": residual
    }

    def test(repeat):
        start = time.perf_counter()

        for i in range(repeat):
            out = model(**inputs)
            torch.cuda.synchronize()

        end = time.perf_counter()

        elapsed_time = end - start

        return elapsed_time

    with memory_profiling(baseline_snapshot,
                          weights_memory=model_memory_usage) as result:

        test(repeat=3)
        elapsed_time = test(repeat=repeat)

    print(f"Batchsize: {n}, "
          f"Throughput: {repeat * n / elapsed_time:.4f} token/s, "
          f"Latency: {elapsed_time*1000 / repeat:.4f} ms. "
          f"weights_memory: {result.weights_memory / GB:.4f}, "
          f"torch_peak_increase: {result.torch_peak_increase / GB:.4f}")


@torch.inference_mode
def test_offloading(repeat):

    with DeviceMemoryProfiler(config.device) as m:
        with torch.device(config.device):
            model, kv_cache, model_memory_usage, baseline_snapshot = init()

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
        process_warp_with_exc(test_prefill, n=bs, repeat=repeat)

    batchsize_list = [2**i for i in range(15)]

    for bs in batchsize_list:
        process_warp_with_exc(test_decoding, bs, repeat)

    process_warp_with_exc(test_offloading, repeat=10)
