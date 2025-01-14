import random
import time
from concurrent.futures import ProcessPoolExecutor

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
        block_allocator=args.block_allocator,
        swap_space=args.swap_space,
        enable_prefix_caching=args.enable_prefix_caching)

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

    for n in range(args.repeat):
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
        actual_hit_rate = np.mean(
            [v[1:][-1] for v in num_cached_tokens.values()]) / args.input_len

        elapsed_time = end - start
        avg_latency = elapsed_time / n_step

        if not args.record_metrics:
            print(f"Batchsize {args.max_num_requests}, Throughput: "
                  f"{len(requests) / elapsed_time:.4f} requests/s, "
                  f"Avg Latency {avg_latency * 1000:0.4f} ms, n_step {n_step}")
        else:

            scheduling_time = np.mean(
                [m.scheduling_time for m in metrics_list])
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


def run(args):
    try:
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark, args)
            f.result()
    except Exception:
        import traceback
        traceback.print_exc()


def test_disable_prefix_caching(args):
    args.swap_space = 0
    args.repeat = 1

    print("test_naive")
    args.block_allocator = "naive"
    run(args)

    # print("test_disable_prefix_caching")
    # args.block_allocator = "disable_prefix_caching"
    # run(args)


def test_prefix_caching(args):
    args.swap_space = 0
    args.repeat = 3

    print("test_prefix_caching")
    args.block_allocator = "prefix_caching"
    run(args)

    # print("test_yoco")
    # args.block_allocator = "yoco"
    # run(args)


def test_offloading(args):
    args.swap_space = 40
    args.repeat = 3

    print("test_offloading+prefix_caching")
    args.block_allocator = "prefix_caching"
    run(args)

    print("test_offloading+no_prefix_caching")
    args.block_allocator = "disable_prefix_caching"
    run(args)


if __name__ == '__main__':
    from easydict import EasyDict as edict

    args = edict()

    args.seed = 0
    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = None

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    args.max_num_requests = 2
    args.max_num_batched_tokens = 1024
    args.record_metrics = True
    args.frieren_executor_max_workers = 1

    args.enable_prefix_caching = None
    args.scheduling = "sync"

    args.input_len = 1024 * 10
    args.output_len = 16
    args.hit_rate = 0.
    """
    # all token in gpu kv cache
    args.num_prompts = 10
    test_disable_prefix_caching(args)
    test_prefix_caching(args)
    test_offloading(args)

    # over gpu kv cacha, offloading play an important role
    args.num_prompts = 20
    test_disable_prefix_caching(args)
    test_prefix_caching(args)
    test_offloading(args)
    """

    for output_len in [2, 4, 8, 16, 32, 64, 128]:
        print("output_len: ", output_len)
        args.num_prompts = 10
        args.output_len = output_len

        test_disable_prefix_caching(args)
        test_prefix_caching(args)
        test_offloading(args)
