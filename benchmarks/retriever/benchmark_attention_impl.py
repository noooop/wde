import os
import random
import time

import numpy as np


def benchmark_wde(args):
    random.seed(args.seed)
    os.environ["WDE_ATTENTION_BACKEND"] = args.attention_impl

    import gc

    import torch

    from wde.tasks.encode_only.arg_utils import \
        EncodeOnlyEngineArgs as EngineArgs
    from wde.workflows.core.llm_engine import LLMEngine

    prompt = "if" * args.input_len
    requests = [prompt for _ in range(args.num_prompts)]

    engine_args = EngineArgs(model=args.model,
                             tokenizer=args.tokenizer,
                             seed=args.seed,
                             trust_remote_code=args.trust_remote_code,
                             dtype=args.dtype,
                             max_model_len=args.max_model_len,
                             device=args.device,
                             max_num_requests=1,
                             scheduling=args.scheduling,
                             record_metrics=args.record_metrics)

    engine = LLMEngine.from_engine_args(engine_args)

    for batchsize in args.batchsize:
        engine.engine_config.scheduler_config.set_args(
            max_num_requests=batchsize)

        start = time.perf_counter()
        metrics_list = []

        for request_id, prompt in enumerate(requests):
            engine.add_request(str(request_id), prompt)

        n_step = 0
        while engine.has_unfinished_requests():
            n_step += 1
            request_outputs = engine.step()
            for request in request_outputs:
                metrics_list.append(request.metrics)

        end = time.perf_counter()

        elapsed_time = end - start
        avg_latency = elapsed_time / n_step

        if metrics_list[0] is None:
            print(
                f"Batchsize {batchsize}, Throughput: "
                f"{len(requests) / elapsed_time:.4f} requests/s, "
                f"Avg Latency {avg_latency * 1000:0.4f} ms, , n_step {n_step}")
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
                f"Batchsize {batchsize}, Throughput: "
                f"{len(requests) / elapsed_time:.4f} requests/s, "
                f"Scheduling time {scheduling_time * 1000:0.4f} ms, "
                f"Num requests {num_requests:.2f}, ",
                f"Num batched tokens {num_batched_tokens:.2f}, ",
                f"Scheduling2inference {scheduling2inference * 1000:0.4f} ms, "
                f"Inference time {inference_time * 1000:0.4f} ms, "
                f"Avg Latency {avg_latency * 1000:0.4f} ms, "
                f"Latency {latency * 1000:0.4f} ms, n_step {n_step}")

        engine.executor.shutdown_execute_loop()
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.num_prompts = 10000

    args.model = 'BAAI/bge-m3'

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.seed = 0
    args.max_model_len = None
    args.device = "cuda"
    args.batchsize = [1, 2, 4, 8, 16, 32, 64]
    args.record_metrics = True

    from concurrent.futures import ProcessPoolExecutor

    from wde.workflows.prefill_only.backends.attention.selector import \
        AttentionImpls

    def run_wde(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_wde, args)
            f.result()

    for scheduling in ["sync", "async"]:
        for dtype, attention_impls in AttentionImpls.items():
            args.scheduling = scheduling
            print("scheduling:", scheduling, "dtype:", dtype)
            for attention_impl in attention_impls:
                print("attention_impl:", attention_impl)
                args.attention_impl = attention_impl
                args.dtype = dtype
                run_wde(args)
