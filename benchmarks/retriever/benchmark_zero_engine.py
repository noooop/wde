import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    from gevent.pool import Pool

    from wde.client import RetrieverClient
    from wde.engine.zero_engine import start_zero_engine

    model_name = args.model

    engine_args = {
        "model": args.model,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "max_num_requests": args.max_num_requests,
        "scheduling": args.scheduling,
        "waiting": args.waiting,
        "record_metrics": args.record_metrics
    }

    server = start_zero_engine(engine_args)

    client = RetrieverClient()

    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)
    print(client.get_services(model_name))

    print("=" * 80)
    print('support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    _prompt = "if" * args.input_len
    requests = [_prompt for _ in range(args.num_prompts)]

    def worker(prompt):
        start = time.perf_counter()
        output = client.encode(name=model_name, inputs=prompt)
        end = time.perf_counter()
        e2e = end - start
        metrics = output.metrics
        metrics["e2e"] = e2e
        metrics = edict(metrics)

        return metrics

    for n_works in args.n_works_list:
        p = Pool(n_works)
        start = time.perf_counter()
        metrics_list = []
        for metrics in p.imap_unordered(worker, requests):
            metrics_list.append(metrics)

        end = time.perf_counter()
        elapsed_time = end - start

        e2e = np.mean([m.e2e for m in metrics_list])

        if not args.record_metrics:
            print(f"n_works {n_works}, Throughput: "
                  f"{len(requests) / elapsed_time:.4f} requests/s, "
                  f"E2E {e2e * 1000:0.4f} ms.")
        else:
            waiting_time = np.mean([m.waiting_time for m in metrics_list])
            scheduling_time = np.mean(
                [m.scheduling_time for m in metrics_list])
            num_requests = np.mean([m.num_requests for m in metrics_list])
            num_batched_tokens = np.mean(
                [m.num_batched_tokens for m in metrics_list])

            scheduling2inference = np.mean(
                [m.scheduling2inference for m in metrics_list])
            inference_time = np.mean([m.inference_time for m in metrics_list])
            latency = np.mean([m.latency for m in metrics_list])

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

    server.terminate()


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

    args.n_works_list = [1, 2, 4, 8, 16, 32, 64, 128]
    args.waiting = None
    args.record_metrics = True

    for max_num_requests in [8, 16]:
        print("max_num_requests:", max_num_requests)
        args.max_num_requests = max_num_requests

        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark, args)
            f.result()
