from benchmarks.offloading_KV_cache.util import get_requests, test
from wde.workflows.decoding.kv_cache.remote.memory import process_warp


def benchmark(args):
    import random
    random.seed(args.seed)

    import wde
    print(wde.__version__)

    requests = get_requests(args)

    try:
        process_warp(test, args, requests)
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    from easydict import EasyDict as edict

    from wde.workflows.decoding.scheduler import BLOCK_ALLOCATOR_MAP

    args = edict()

    args.input_len = 8192
    args.output_len = 1
    args.num_prompts = 4

    args.seed = 0
    args.model = "Qwen/Qwen2.5-3B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
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

    def test_vary_max_num_batched_tokens(args):
        max_num_batched_tokens_list = [1024, 768, 512, 384, 256, 128, 64, 32]

        for max_num_batched_tokens in max_num_batched_tokens_list:
            args.max_num_batched_tokens = max_num_batched_tokens
            process_warp(benchmark, args)

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
        for block_allocator in list(BLOCK_ALLOCATOR_MAP.keys()):
            print("block_allocator", block_allocator)
            args.block_allocator = block_allocator
            args.enable_prefix_caching = None
            test_vary_scheduling(args)

        args.swap_space = 40
        for enable_prefix_caching in [True, False]:
            print("kv_cache_manager", "OffloadingKVCaching",
                  "enable_prefix_caching", enable_prefix_caching)
            args.block_allocator = None
            args.enable_prefix_caching = enable_prefix_caching
            test_vary_scheduling(args)

    test_vary_block_allocator(args)
