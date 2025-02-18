import time

import numpy as np


def test(args, requests):
    from wde import LLMEngine, SamplingParams
    from wde.workflows.core.schema.engine_io import TokensPrompt
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs

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
        block_allocator=args.block_allocator,
        enable_prefix_caching=args.enable_prefix_caching,
        swap_space=args.swap_space,
        remote_kv_cache_server=args.remote_kv_cache_server,
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
            print(
                f"num_batched_tokens {args.max_num_batched_tokens}, Throughput: "
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
                f"num_batched_tokens {args.max_num_batched_tokens}, Throughput: "
                f"{len(requests) / elapsed_time:.4f} requests/s, "
                f"actual hit rate {actual_hit_rate}, "
                f"Scheduling time {scheduling_time * 1000:0.4f} ms, "
                f"Num requests {num_requests:.2f}, ",
                f"Num batched tokens {num_batched_tokens:.2f}, ",
                f"Scheduling2inference {scheduling2inference * 1000:0.4f} ms, "
                f"Inference time {inference_time * 1000:0.4f} ms, "
                f"Avg Latency {avg_latency * 1000:0.4f} ms, "
                f"Latency {latency * 1000:0.4f} ms, n_step {n_step}")

    engine.join()
    del engine


def start_remote_kv_cache(args):
    from wde.microservices.framework.zero.server import ZeroServerProcess
    from wde.microservices.standalone.server import setup_and_run

    remote_server_class = "wde.workflows.decoding.kv_cache_server.server:ZeroRemoteKVCacheServer"
    server = setup_and_run()
    try:
        kv_cache_server = ZeroServerProcess(server_class=remote_server_class,
                                            server_kwargs={
                                                "model":
                                                args.model,
                                                "name":
                                                args.get("server_name", None),
                                                "block_size":
                                                args.block_size,
                                                "memory_space":
                                                args.memory_space,
                                                "cache_dtype":
                                                args.cache_dtype
                                            })

        kv_cache_server.start()
    except Exception as e:
        server.terminate()
        raise e

    return [server, kv_cache_server]


def wait_service_available(args):
    from wde.workflows.decoding.kv_cache_server.client import \
        ZeroRemoteKVCacheClient

    client = ZeroRemoteKVCacheClient()
    client.wait_service_available(args.remote_kv_cache_server_name)


def kv_cache_info(args):
    from wde.workflows.decoding.kv_cache_server.client import \
        ZeroRemoteKVCacheClient

    client = ZeroRemoteKVCacheClient()
    print(client.info(args.remote_kv_cache_server_name))
