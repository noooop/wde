from benchmarks.offloading_KV_cache.util import get_requests
from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             start_remote_kv_cache, test,
                                             wait_service_available)
from wde.utils import process_warp_with_exc
from wde.workflows.decoding.scheduler import BLOCK_ALLOCATOR_MAP


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

    for s in server:
        s.terminate()


if __name__ == '__main__':

    from easydict import EasyDict as edict

    args = edict()

    args.input_len = 8192
    args.output_len = 1
    args.num_prompts = 4

    args.seed = 0
    args.model = "Qwen/Qwen2.5-3B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.device = "cuda"

    args.max_model_len = 10000

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    args.max_num_requests = 1
    args.record_metrics = True
    args.frieren_executor_max_workers = 1
    args.hit_rate = 0.

    args.block_size = 16
    args.cache_dtype = "auto"
    args.remote_kv_cache_server = True

    def test_vary_max_num_batched_tokens(args):
        max_num_batched_tokens_list = [1024, 768, 512, 384, 256, 128, 64, 32]

        for max_num_batched_tokens in max_num_batched_tokens_list:
            args.max_num_batched_tokens = max_num_batched_tokens
            process_warp_with_exc(benchmark, args)

    def test_vary_scheduling(args):
        for scheduling in ["sync"]:
            print(f"scheduling: {scheduling}")
            args.scheduling = scheduling
            print()

            test_vary_max_num_batched_tokens(args)

        for scheduling in ["async"]:
            for max_workers in [2]:
                print(f"scheduling: {scheduling}-{max_workers}")
                args.frieren_executor_max_workers = max_workers
                args.scheduling = scheduling
                print()

                test_vary_max_num_batched_tokens(args)

    def test_vary_block_allocator(args):
        args.swap_space = 0
        args.memory_space = 0
        for block_allocator in list(BLOCK_ALLOCATOR_MAP.keys()):
            print("block_allocator", block_allocator)
            args.block_allocator = block_allocator
            # test_vary_scheduling(args)

        args.swap_space = 40
        args.memory_space = 40
        for enable_prefix_caching in [True, False]:
            print("kv_cache_manager", "remote_kv_cache",
                  "enable_prefix_caching", enable_prefix_caching)
            args.block_allocator = None
            args.enable_prefix_caching = enable_prefix_caching
            test_vary_scheduling(args)

    test_vary_block_allocator(args)
