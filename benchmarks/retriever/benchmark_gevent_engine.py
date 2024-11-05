import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import shortuuid


def benchmark(args):
    random.seed(args.seed)

    from gevent.pool import Pool

    from wde.engine.gevent_engine import GeventLLMEngine

    engine = GeventLLMEngine(model=args.model,
                             seed=args.seed,
                             dtype=args.dtype,
                             device=args.device,
                             max_num_requests=args.max_num_requests,
                             scheduling=args.scheduling,
                             waiting=args.waiting)

    _prompt = "if" * args.input_len
    requests = [_prompt for _ in range(args.num_prompts)]

    def worker(prompt):
        start = time.perf_counter()
        request_id = f"{shortuuid.random(length=22)}"
        outputs = engine.encode(inputs=prompt, request_id=request_id)
        output = list(outputs)[0]
        end = time.perf_counter()
        e2e = end - start

        m = output.metrics

        metrics = edict({
            "waiting_time": m.waiting_time,
            "scheduling_time": m.scheduling_time,
            "num_requests": m.num_requests,
            "num_batched_tokens": m.num_batched_tokens,
            "scheduling2inference": m.scheduling2inference,
            "inference_time": m.inference_time,
            "latency": m.latency,
            "e2e": e2e,
        })
        return metrics

    for n_works in args.n_works_list:
        p = Pool(n_works)
        start = time.perf_counter()

        metrics_list = []
        for metrics in p.imap_unordered(worker, requests):
            metrics_list.append(metrics)

        end = time.perf_counter()
        elapsed_time = end - start
        waiting_time = np.mean([m.waiting_time for m in metrics_list])
        scheduling_time = np.mean([m.scheduling_time for m in metrics_list])
        num_requests = np.mean([m.num_requests for m in metrics_list])
        num_batched_tokens = np.mean(
            [m.num_batched_tokens for m in metrics_list])

        scheduling2inference = np.mean(
            [m.scheduling2inference for m in metrics_list])
        inference_time = np.mean([m.inference_time for m in metrics_list])
        latency = np.mean([m.latency for m in metrics_list])
        e2e = np.mean([m.e2e for m in metrics_list])

        overhead = e2e - latency - waiting_time

        print(
            f"n_works {n_works}, Throughput: "
            f"{len(requests) / elapsed_time:.4f} requests/s, "
            f"Scheduling time {scheduling_time * 1000:0.4f} ms, "
            f"Num requests {num_requests:.2f}, ",
            f"Num batched tokens {num_batched_tokens:.2f}, ",
            f"Scheduling2inference {scheduling2inference * 1000:0.4f} ms, "
            f"Inference time {inference_time * 1000:0.4f} ms, "
            f"Waiting time {waiting_time * 1000:0.4f} ms, "
            f"Latency {latency * 1000:0.4f} ms, "
            f"E2E {e2e * 1000:0.4f} ms, "
            f"Overhead {overhead * 1000:0.4f} ms.")


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.num_prompts = 10000

    args.model = 'BAAI/bge-m3'
    args.seed = 0
    args.dtype = "half"
    args.device = "cuda"
    args.scheduling = "async"
    args.waiting = None

    args.n_works_list = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    for max_num_requests in [1, 2, 4, 8, 16, 32, 64]:
        print("max_num_requests:", max_num_requests)
        args.max_num_requests = max_num_requests

        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark, args)
            f.result()
