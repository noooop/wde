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
            "scheduler_time": m.scheduler_time,
            "n_request_in_batch": m.n_request_in_batch,
            "waiting4execution": m.waiting4execution,
            "execute_time": m.execute_time,
            "delay": m.delay,
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
        scheduler_time = np.mean([m.scheduler_time for m in metrics_list])
        waiting4execution = np.mean(
            [m.waiting4execution for m in metrics_list])
        execute_time = np.mean([m.execute_time for m in metrics_list])
        n_request_in_batch = np.mean(
            [m.n_request_in_batch for m in metrics_list])
        delay = np.mean([m.delay for m in metrics_list])
        e2e = np.mean([m.e2e for m in metrics_list])

        print(
            f"n_works {n_works}, Throughput: "
            f"{len(requests) / elapsed_time:.4f} requests/s, "
            f"Actual batchsize {n_request_in_batch:.2f}, ",
            f"Waiting time {waiting_time * 1000:0.4f} ms, "
            f"Scheduler time {scheduler_time * 1000:0.4f} ms, "
            f"Waiting for Execute {waiting4execution * 1000:0.4f} ms, "
            f"Execute time {execute_time * 1000:0.4f} ms, "
            f"Delay {delay * 1000:0.4f} ms, "
            f"E2E {e2e * 1000:0.4f} ms")


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
