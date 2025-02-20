import random
import time

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    import wde
    print(wde.__version__)

    from benchmarks.offloading_KV_cache.util import get_requests
    from wde import LLMEngine, SamplingParams
    from wde.workflows.core.schema.engine_io import TokensPrompt
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs

    requests = get_requests(args)

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_requests=args.max_num_requests,
        scheduling=args.scheduling,
        frieren_executor_max_workers=args.frieren_executor_max_workers,
        record_metrics=args.record_metrics,
        block_size=args.block_size)

    engine = LLMEngine.from_engine_args(engine_args)

    for n in range(args.get("repeat", 1)):
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
