import os
import random
import time

import numpy as np


def benchmark_wde(args):
    random.seed(args.seed)
    os.environ["WDE_ATTENTION_BACKEND"] = args.attention_impl

    import gc

    import torch

    from wde.tasks.core.llm_engine import LLMEngine
    from wde.tasks.encode_only.arg_utils import \
        EncodeOnlyEngineArgs as EngineArgs

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
                             scheduling=args.scheduling)

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

        scheduler_time = np.mean([m.scheduler_time for m in metrics_list])
        waiting4execution = np.mean(
            [m.waiting4execution for m in metrics_list])
        execute_time = np.mean([m.execute_time for m in metrics_list])
        n_request_in_batch = np.mean(
            [m.n_request_in_batch for m in metrics_list])
        delay = np.mean([m.delay for m in metrics_list])
        avg_delay = elapsed_time / n_step

        print(
            f"Batchsize {batchsize}, Throughput: "
            f"{len(requests) / elapsed_time:.4f} requests/s, "
            f"Actual batchsize {n_request_in_batch:.2f}, ",
            f"Scheduler time {scheduler_time * 1000:0.4f} ms, "
            f"Waiting for Execute {waiting4execution * 1000:0.4f} ms, "
            f"Execute time {execute_time * 1000:0.4f} ms, "
            f"Avg Delay {avg_delay * 1000:0.4f} ms, "
            f"Delay {delay * 1000:0.4f} ms, n_step {n_step}")

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

    from concurrent.futures import ProcessPoolExecutor

    from wde.tasks.prefill_only.backends.attention.selector import \
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
