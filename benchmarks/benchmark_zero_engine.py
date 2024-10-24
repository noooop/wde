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
        "waiting": args.waiting
    }

    handle = start_zero_engine(engine_args)

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
        delay = end - start
        return output, delay

    for n_works in args.n_works_list:
        p = Pool(n_works)
        start = time.perf_counter()
        delay_list = []
        for output, delay in p.imap_unordered(worker, requests):
            delay_list.append(delay)

        end = time.perf_counter()
        elapsed_time = end - start

        print(f"n_works {n_works}, Throughput: "
              f"{len(requests) / elapsed_time:.4f} requests/s, "
              f"Delay {np.mean(delay_list) * 1000:0.2f} ms")

    for h in handle:
        h.terminate()


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

    for waiting in [None, 0.001]:
        print("waiting", waiting)
        args.waiting = waiting
        for max_model_len in [1, 2, 4, 8, 16, 32, 64]:
            print("max_model_len:", max_model_len)
            args.max_model_len = max_model_len

            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
