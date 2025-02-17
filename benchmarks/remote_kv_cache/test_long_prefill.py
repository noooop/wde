from benchmarks.offloading_KV_cache.util import get_requests
from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             start_remote_kv_cache, test,
                                             wait_service_available)
from wde.workflows.decoding.kv_cache.remote.memory import process_warp


def benchmark(args):
    import random
    random.seed(args.seed)

    import wde
    print(wde.__version__)

    server = start_remote_kv_cache(args)
    process_warp(wait_service_available, args)

    requests = get_requests(args)

    try:
        process_warp(kv_cache_info, args)
        process_warp(test, args, requests)
        process_warp(kv_cache_info, args)
        process_warp(test, args, requests)
        process_warp(kv_cache_info, args)
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        for s in server:
            s.terminate()


def test_remote(args):
    args.remote_kv_cache_server_name = "kv_cache_server"
    args.swap_space = 40
    args.repeat = 3

    print("test_remote+prefix_caching")
    args.enable_prefix_caching = True
    process_warp(benchmark, args)

    print("test_remote+no_prefix_caching")
    args.enable_prefix_caching = False
    process_warp(benchmark, args)


if __name__ == '__main__':
    from easydict import EasyDict as edict

    args = edict()

    args.seed = 0
    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
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
    args.output_len = 16
    args.hit_rate = 0.
    args.block_size = 16

    for output_len in [2, 4, 8, 16, 32, 64, 128]:
        print("output_len: ", output_len)
        args.num_prompts = 10
        args.output_len = output_len

        test_remote(args)
