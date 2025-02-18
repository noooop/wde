from benchmarks.offloading_KV_cache.util import get_requests
from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             start_remote_kv_cache, test,
                                             wait_service_available)
from wde.utils import process_warp_with_exc


def benchmark(args):
    import random
    random.seed(args.seed)

    import wde
    print(wde.__version__)

    server = start_remote_kv_cache(args)

    args.remote_kv_cache_server_name = f"kv_cache:{args.model}:{args.block_size}"

    process_warp_with_exc(wait_service_available, args)

    requests = get_requests(args)

    process_warp_with_exc(kv_cache_info, args)
    process_warp_with_exc(test, args, requests)
    process_warp_with_exc(kv_cache_info, args)
    process_warp_with_exc(test, args, requests)
    process_warp_with_exc(kv_cache_info, args)

    for s in server:
        s.terminate()


def test_remote(args):
    args.remote_kv_cache_server_name = "kv_cache_server"
    args.swap_space = 40
    args.repeat = 3

    print("test_remote+prefix_caching")
    args.enable_prefix_caching = True
    process_warp_with_exc(benchmark, args)

    print("test_remote+no_prefix_caching")
    args.enable_prefix_caching = False
    process_warp_with_exc(benchmark, args)


if __name__ == '__main__':
    from easydict import EasyDict as edict

    args = edict()

    args.seed = 0
    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = None

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    args.max_num_requests = 2
    args.max_num_batched_tokens = 1024
    args.record_metrics = True
    args.frieren_executor_max_workers = 1
    args.block_allocator = None
    args.scheduling = "sync"

    args.input_len = 1024 * 10
    args.hit_rate = 0.
    args.block_size = 16
    args.swap_space = 40
    args.memory_space = 40
    args.remote_kv_cache_server = True

    for output_len in [2, 4, 8, 16, 32, 64, 128]:
        print("output_len: ", output_len)
        args.num_prompts = 10
        args.output_len = output_len

        test_remote(args)
