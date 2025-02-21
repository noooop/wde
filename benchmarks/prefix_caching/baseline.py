from benchmarks.chat.util import get_requests
from benchmarks.prefix_caching.util import test
from wde.utils import process_warp_with_exc


def benchmark(args):
    requests = get_requests(args)
    process_warp_with_exc(test, args, requests)


if __name__ == '__main__':
    from easydict import EasyDict as edict

    from wde.workflows.decoding.scheduler import BLOCK_ALLOCATOR_MAP

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

    args.max_num_requests = 32
    args.max_num_batched_tokens = 1024
    args.record_metrics = True
    args.frieren_executor_max_workers = 1

    def test_vary_hit_rate(args):
        for hit_rate in [0.1 * x for x in range(0, 11)]:
            args.hit_rate = hit_rate
            process_warp_with_exc(benchmark, args)

    def vary_block_allocator(args):
        for block_allocator in list(BLOCK_ALLOCATOR_MAP.keys()):
            print("block_allocator", block_allocator)
            args.block_allocator = block_allocator

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

    vary_block_allocator(args)
