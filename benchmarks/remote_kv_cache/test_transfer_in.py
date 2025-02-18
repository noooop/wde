import random

from benchmarks.offloading_KV_cache.util import get_requests
from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             process_warp_with_exc,
                                             start_remote_kv_cache, test,
                                             wait_service_available)


def benchmark(args):
    random.seed(args.seed)

    import wde
    print(wde.__version__)

    server = start_remote_kv_cache(args)

    args.remote_kv_cache_server_name = f"kv_cache:{args.model}:{args.block_size}"

    process_warp_with_exc(wait_service_available, args)

    requests = get_requests(args)

    try:
        process_warp_with_exc(kv_cache_info, args)
        process_warp_with_exc(test, args, requests)
        process_warp_with_exc(kv_cache_info, args)
        process_warp_with_exc(test, args, requests)
        process_warp_with_exc(kv_cache_info, args)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        for s in server:
            s.terminate()


if __name__ == '__main__':
    from easydict import EasyDict as edict

    args = edict()

    args.input_len = 1000
    args.output_len = 24
    args.num_prompts = 1000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-3B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 2000

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9
    args.record_metrics = True
    args.frieren_executor_max_workers = 1

    args.block_size = 16

    args.max_num_requests = 32
    args.max_num_batched_tokens = 1024

    args.swap_space = 40
    args.memory_space = 40
    args.cache_dtype = "auto"
    args.remote_kv_cache_server = True
    args.block_allocator = None

    def test_vary_hit_rate(args):
        for hit_rate in [0.1 * x for x in range(0, 11)]:
            args.hit_rate = hit_rate
            process_warp_with_exc(benchmark, args)

    def test_vary_scheduling(args):
        for scheduling in ["sync"]:
            print(f"scheduling: {scheduling}")
            args.scheduling = scheduling
            print()

            test_vary_hit_rate(args)

        for scheduling in ["async"]:
            for max_workers in [2]:
                print(f"scheduling: {scheduling}-{max_workers}")
                args.frieren_executor_max_workers = max_workers
                args.scheduling = scheduling
                print()

                test_vary_hit_rate(args)

    def test_vary_enable_prefix_caching(args):
        for enable_prefix_caching in [True, False]:
            print("enable_prefix_caching: ", enable_prefix_caching)
            args.enable_prefix_caching = enable_prefix_caching
            test_vary_scheduling(args)

    test_vary_enable_prefix_caching(args)
