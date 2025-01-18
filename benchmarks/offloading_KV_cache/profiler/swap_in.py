import random
import time

import numpy as np
import torch


def benchmark(args):
    random.seed(args.seed)

    import wde
    from wde import LLMEngine, SamplingParams
    from wde.workflows.core.schema.engine_io import TokensPrompt
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs
    from wde.workflows.decoding.backends.sampling.utils import TokenSampler

    print(wde.__version__)

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_requests=args.max_num_requests,
        scheduling=args.scheduling,
        frieren_executor_max_workers=args.frieren_executor_max_workers,
        record_metrics=args.record_metrics,
        block_allocator=args.block_allocator,
        swap_space=args.swap_space)

    engine = LLMEngine.from_engine_args(engine_args)
    token_sampler = TokenSampler(args.tokenizer)

    prefix_len = int(args.hit_rate * args.input_len)
    unique_len = args.input_len - prefix_len
    prefix_token_ids = token_sampler.random_sample(prefix_len)

    requests = []
    for _ in range(args.num_prompts):
        unique_part_token_ids = token_sampler.random_sample(unique_len)

        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        requests.append(prompt_token_ids)

    start = time.perf_counter()
    metrics_list = []

    for request_id, prompt_token_ids in enumerate(requests):
        inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            ignore_eos=True,
            max_tokens=args.output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    num_cached_tokens_dict = {}

    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        n_step = 0
        while engine.has_unfinished_requests():
            n_step += 1

            request_outputs = engine.step()
            for request in request_outputs:
                metrics_list.append(request.metrics)

                request_id = request.request_id

                if request_id not in num_cached_tokens_dict:
                    num_cached_tokens_dict[request_id] = []

                num_cached_tokens_dict[request_id].append(
                    request.num_cached_tokens)

    end = time.perf_counter()

    prof.export_chrome_trace("test_swap_in.json")

    num_cached_tokens = np.array(
        [0 for i in range(len(num_cached_tokens_dict))])
    for k, v in num_cached_tokens_dict.items():
        num_cached_tokens[int(k)] = v[-1]
        print(k, v)

    actual_hit_rate = np.mean(num_cached_tokens / args.input_len)

    elapsed_time = end - start
    avg_latency = elapsed_time / n_step

    if not args.record_metrics:
        print(f"Batchsize {args.max_num_requests}, Throughput: "
              f"{len(requests) / elapsed_time:.4f} requests/s, "
              f"Avg Latency {avg_latency * 1000:0.4f} ms, n_step {n_step}")
    else:

        scheduling_time = np.mean([m.scheduling_time for m in metrics_list])
        num_requests = np.mean([m.num_requests for m in metrics_list])
        num_batched_tokens = np.mean(
            [m.num_batched_tokens for m in metrics_list])

        scheduling2inference = np.mean(
            [m.scheduling2inference for m in metrics_list])
        inference_time = np.mean([m.inference_time for m in metrics_list])
        latency = np.mean([m.latency for m in metrics_list])
        print(
            f"Batchsize {args.max_num_requests}, Throughput: "
            f"{len(requests) / elapsed_time:.4f} requests/s, "
            f"actual hit rate {actual_hit_rate}, "
            f"Scheduling time {scheduling_time * 1000:0.4f} ms, "
            f"Num requests {num_requests:.2f}, ",
            f"Num batched tokens {num_batched_tokens:.2f}, ",
            f"Scheduling2inference {scheduling2inference * 1000:0.4f} ms, "
            f"Inference time {inference_time * 1000:0.4f} ms, "
            f"Avg Latency {avg_latency * 1000:0.4f} ms, "
            f"Latency {latency * 1000:0.4f} ms, n_step {n_step}")


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    from easydict import EasyDict as edict

    args = edict()

    args.input_len = 1024 * 2
    args.output_len = 16
    args.num_prompts = 4

    args.seed = 0
    args.model = "Qwen/Qwen2.5-3B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 10000

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    args.max_num_requests = 32
    args.max_num_batched_tokens = 1024
    args.record_metrics = True
    args.frieren_executor_max_workers = 1

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    args.swap_space = 10
    args.scheduling = "sync"
    args.block_allocator = None
    args.enable_prefix_caching = False
    args.hit_rate = 1.

    benchmark(args)
