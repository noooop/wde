import random
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    from gevent.pool import Pool

    from wde.engine.zero_engine import start_zero_engine
    from wde.tasks.retriever.engine.client import RetrieverClient

    model_name = args.model

    engine_args = {
        "model": args.model,
        "seed": args.seed,
        "dtype": args.dtype,
        "device": args.device,
        "max_num_seqs": args.max_model_len,
        "scheduling": args.scheduling,
        "waiting": args.waiting,
        "return_metrics": True
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

    for waiting in [0.001, 0.005, 0.01]:
        print("waiting", waiting)
        args.waiting = waiting
        for max_model_len in [1, 2, 4, 8, 16, 32, 64]:
            print("max_model_len:", max_model_len)
            args.max_model_len = max_model_len

            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
