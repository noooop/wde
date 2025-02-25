from benchmarks.prefix_caching.baseline_vllm import benchmark
from wde.utils import process_warp_with_exc

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

    args.enable_chunked_prefill = True
    args.max_num_requests = 2
    args.max_num_batched_tokens = 1024
    args.tensor_parallel_size = 1

    args.input_len = 1024 * 10
    args.num_prompts = 10
    args.hit_rate = 0.
    args.repeat = 3

    args.max_num_seqs = args.max_num_requests

    def test_vary_enable_prefix_caching(args):
        for enable_prefix_caching in [False, True]:
            args.enable_prefix_caching = enable_prefix_caching

            process_warp_with_exc(benchmark, args)

    def test_vary_enforce_eager(args):
        for enforce_eager in [False, True]:
            args.enforce_eager = enforce_eager

            test_vary_enable_prefix_caching(args)

    def test_v0_v1(args):
        for use_v1 in [False, True]:
            args.use_v1 = use_v1
            test_vary_enforce_eager(args)

    for output_len in [2, 4, 8, 16, 32, 64, 128]:
        args.output_len = output_len
        test_v0_v1(args)
