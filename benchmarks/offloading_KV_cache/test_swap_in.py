from benchmarks.chat.util import get_requests
from benchmarks.offloading_KV_cache.util import test
from wde.utils import process_warp_with_exc


def benchmark(args):
    requests = get_requests(args)
    process_warp_with_exc(test, args, requests)


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

    args.max_num_requests = 32
    args.max_num_batched_tokens = 1024

    args.swap_space = 40
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
