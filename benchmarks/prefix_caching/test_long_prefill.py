from benchmarks.chat.util import get_requests
from benchmarks.prefix_caching.util import test
from wde.utils import process_warp_with_exc


def benchmark(args):
    requests = get_requests(args)
    process_warp_with_exc(test, args, requests)


def test_disable_prefix_caching(args):
    args.swap_space = 0
    args.repeat = 1

    print("test_naive")
    args.block_allocator = "naive"
    process_warp_with_exc(benchmark, args)

    print("test_disable_prefix_caching")
    args.block_allocator = "disable_prefix_caching"
    process_warp_with_exc(benchmark, args)


def test_prefix_caching(args):
    args.swap_space = 0
    args.repeat = 3

    print("test_prefix_caching")
    args.block_allocator = "prefix_caching"
    process_warp_with_exc(benchmark, args)

    print("test_yoco")
    args.block_allocator = "yoco"
    process_warp_with_exc(benchmark, args)


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

    args.enable_prefix_caching = None
    args.scheduling = "sync"

    args.input_len = 1024 * 10
    args.output_len = 16
    args.hit_rate = 0.
    """
    # all token in gpu kv cache
    args.num_prompts = 10
    test_disable_prefix_caching(args)
    test_prefix_caching(args)
    test_offloading(args)

    # over gpu kv cacha, offloading play an important role
    args.num_prompts = 20
    test_disable_prefix_caching(args)
    test_prefix_caching(args)
    test_offloading(args)
    """

    for output_len in [2, 4, 8, 16, 32, 64, 128]:
        print("output_len: ", output_len)
        args.num_prompts = 10
        args.output_len = output_len

        test_disable_prefix_caching(args)
        test_prefix_caching(args)
