# ruff: noqa: F841, E402

import gc
import time

import torch
import vllm.forward_context as forward_context
from vllm.attention import Attention
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.utils import DeviceMemoryProfiler, MemorySnapshot, memory_profiling

from .util import GB, cache_config, config, quant_config


def init():
    kv_cache = torch.rand(2,
                          config.n_blocks,
                          config.block_size,
                          config.num_attention_heads,
                          config.qk_head_dim,
                          dtype=config.dtype,
                          device=config.device)

    with DeviceMemoryProfiler(config.device) as m:
        attn = Attention(num_heads=config.num_attention_heads,
                         head_size=config.qk_head_dim,
                         scale=config.scaling,
                         num_kv_heads=config.num_attention_heads,
                         cache_config=cache_config,
                         quant_config=quant_config)
        attn.use_output = True
        attn.kv_cache = [kv_cache]

    model_memory_usage = m.consumed_memory

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    baseline_snapshot = MemorySnapshot()

    return attn, kv_cache, model_memory_usage, baseline_snapshot


@torch.inference_mode
def test_prefill(n, repeat, use_direct_call):
    q = torch.rand(n,
                   config.num_attention_heads,
                   config.qk_head_dim,
                   dtype=config.dtype,
                   device=config.device)
    k = torch.rand(n,
                   config.num_attention_heads,
                   config.qk_head_dim,
                   dtype=config.dtype,
                   device=config.device)
    v = torch.rand(n,
                   config.num_attention_heads,
                   config.qk_head_dim,
                   dtype=config.dtype,
                   device=config.device)

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

    attn, kv_cache, model_memory_usage, baseline_snapshot = init()
    attn.use_direct_call = use_direct_call

    forward_context._forward_context = forward_context.ForwardContext(
        attn_layers={"": attn}, attn_metadata=attn_metadata, virtual_engine=0)

    def test(repeat):
        start = time.perf_counter()

        for i in range(repeat):
            attn_output = attn(q, k, v, kv_cache, attn_metadata)
            assert attn_output.shape == (n * config.num_attention_heads,
                                         config.qk_head_dim)
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
def test_decoding(n, repeat, use_direct_call):
    q = torch.rand(1,
                   config.num_attention_heads,
                   config.qk_head_dim,
                   dtype=config.dtype,
                   device=config.device)
    k = torch.rand(1,
                   config.num_attention_heads,
                   config.qk_head_dim,
                   dtype=config.dtype,
                   device=config.device)
    v = torch.rand(1,
                   config.num_attention_heads,
                   config.qk_head_dim,
                   dtype=config.dtype,
                   device=config.device)

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

    attn, kv_cache, model_memory_usage, baseline_snapshot = init()
    attn.use_direct_call = use_direct_call

    forward_context._forward_context = forward_context.ForwardContext(
        attn_layers={"": attn}, attn_metadata=attn_metadata, virtual_engine=0)

    def test(repeat):
        start = time.perf_counter()

        for i in range(repeat):
            attn_output = attn(q, k, v, kv_cache, attn_metadata)
            assert attn_output.shape == (1 * config.num_attention_heads,
                                         config.qk_head_dim)
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


if __name__ == '__main__':
    from wde.utils import process_warp_with_exc
    repeat = 1000

    batchsize_list = [2**i for i in range(14)]
    """
    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        test_prefill(n=1024, repeat=10)
    prof.export_chrome_trace(f"attn.json")
    """

    for bs in batchsize_list:
        process_warp_with_exc(test_prefill, bs, repeat, use_direct_call=True)

    for bs in batchsize_list:
        process_warp_with_exc(test_prefill, bs, repeat, use_direct_call=False)

    batchsize_list = [2**i for i in range(15)]

    for bs in batchsize_list:
        process_warp_with_exc(test_decoding, bs, repeat, use_direct_call=True)

    for bs in batchsize_list:
        process_warp_with_exc(test_decoding, bs, repeat, use_direct_call=False)
