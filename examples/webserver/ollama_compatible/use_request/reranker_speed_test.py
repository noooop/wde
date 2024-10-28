import time

import numpy as np
import requests
from easydict import EasyDict as edict
from gevent import monkey
from gevent.pool import Pool

monkey.patch_socket()


def worker(pairs):
    start = time.perf_counter()
    response = requests.post('http://localhost:11434/api/reranker',
                             json={
                                 "model": "BAAI/bge-reranker-v2-m3",
                                 "query": pairs[0],
                                 "passage": pairs[1],
                             })

    score = response.json()["score"]
    end = time.perf_counter()
    e2e = end - start
    metrics = edict({"e2e": e2e})
    return score, metrics


def speed_test(args):

    _prompt = "if" * args.input_len
    pairs_list = [(_prompt, _prompt) for _ in range(args.num_prompts)]

    for n_works in args.n_works_list:
        p = Pool(n_works)
        start = time.perf_counter()

        score_list = []
        metrics_list = []
        for score, metrics in p.imap_unordered(worker, pairs_list):
            metrics_list.append(metrics)
            score_list.append(score)

        end = time.perf_counter()
        elapsed_time = end - start

        e2e = np.mean([m.e2e for m in metrics_list])

        print(f"n_works {n_works}, Throughput: "
              f"{len(pairs_list) / elapsed_time:.4f} requests/s, "
              f"E2E {e2e * 1000:0.4f} ms")


if __name__ == '__main__':
    args = edict()

    args.model = "BAAI/bge-reranker-v2-m3"

    args.input_len = 256
    args.num_prompts = 1000
    args.n_works_list = [1, 2, 4, 8, 16, 32, 64, 128]

    speed_test(args)
