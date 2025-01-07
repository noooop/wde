import random
import time

import numpy as np


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
        block_allocator=args.block_allocator)

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

    num_cached_tokens = {}

    n_step = 0
    while engine.has_unfinished_requests():
        n_step += 1
        request_outputs = engine.step()
        for request in request_outputs:
            metrics_list.append(request.metrics)

            request_id = request.request_id

            if request_id not in num_cached_tokens:
                num_cached_tokens[request_id] = []

            num_cached_tokens[request_id].append(request.num_cached_tokens)

    end = time.perf_counter()
    actual_hit_rate = np.mean([v[1:][-1] for v in num_cached_tokens.values()
                               ]) / args.input_len

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

    from wde.workflows.decoding.scheduler import BLOCK_ALLOCATOR_MAP

    args = edict()

    args.input_len = 1000
    args.output_len = 24
    args.num_prompts = 1000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-3B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 2000

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

    for block_allocator in list(BLOCK_ALLOCATOR_MAP.keys()):
        print("block_allocator", block_allocator)
        args.block_allocator = block_allocator

        for scheduling in ["sync", "simple_async"]:
            print(f"scheduling: {scheduling}")
            args.scheduling = scheduling
            print()

            for hit_rate in [0.1 * x for x in range(0, 11)]:
                args.hit_rate = hit_rate
                run(args)

        for scheduling in ["async"]:
            for max_workers in [1, 2, 3]:
                print(f"scheduling: {scheduling}-{max_workers}")
                args.frieren_executor_max_workers = max_workers
                args.scheduling = scheduling
                print()

                for hit_rate in [0.1 * x for x in range(0, 11)]:
                    args.hit_rate = hit_rate
                    run(args)