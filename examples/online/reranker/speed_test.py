import sys
import time

import numpy as np
from easydict import EasyDict as edict
from gevent.pool import Pool

from wde import const, envs
from wde.microservices.framework.nameserver.client import NameServerClient
from wde.microservices.framework.zero.schema import Timeout
from wde.microservices.framework.zero_manager.client import ZeroManagerClient
from wde.tasks.reranker.engine.client import RerankerClient

nameserver_client = NameServerClient()

try:
    nameserver_client.support_methods()
except Timeout:
    print("Failed to connect to server.\n"
          "Need to start server in another console.\n"
          "python -m wde server")
    sys.exit()


def speed_test(args):
    #########################################################

    model_name = args.model

    engine_args = {"model": model_name}

    #########################################################
    # Start engine
    manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
    manager_client.wait_service_available(envs.ROOT_MANAGER_NAME)

    model_name = engine_args["model"]

    out = manager_client.start(name=model_name,
                               engine_kwargs={
                                   "server_class":
                                   const.INFERENCE_ENGINE_CLASS,
                                   "engine_args": engine_args
                               })
    print("Start engine:", out)

    ###############################################################
    # Wait until ready to use

    client = RerankerClient()

    print("=" * 80)
    print(f"Wait {model_name} available")
    client.wait_service_available(model_name)

    ###############################################################
    # Query basic information

    print(client.get_services(model_name))

    print("=" * 80)
    print('support_methods')
    print(client.support_methods(model_name))
    print(client.info(model_name))

    ###############################################################
    # encode

    _prompt = "if" * args.input_len
    requests = [(_prompt, _prompt) for _ in range(args.num_prompts)]

    def worker(pairs):
        start = time.perf_counter()
        client.compute_score(name=model_name, pairs=pairs)
        end = time.perf_counter()
        e2e = end - start
        metrics = edict({"e2e": e2e})
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

        print(f"n_works {n_works}, Throughput: "
              f"{len(requests) / elapsed_time:.4f} requests/s, "
              f"E2E {e2e * 1000:0.4f} ms")

    ###############################################################
    # Terminate engine

    out = manager_client.terminate(name=model_name)
    print("Terminate engine:", out)


if __name__ == '__main__':
    args = edict()

    args.model = "BAAI/bge-reranker-v2-m3"
    args.dtype = "half"
    args.device = "cuda"

    args.input_len = 256
    args.num_prompts = 10000
    args.n_works_list = [1, 2, 4, 8, 16, 32, 64, 128]

    speed_test(args)
