from benchmarks.chat.util import benchmark
from wde.utils import process_warp_with_exc

if __name__ == '__main__':
    from easydict import EasyDict as edict

    args = edict()

    args.seed = 0
    args.quantization = None
    args.dtype = 'auto'
    args.cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = None

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.gpu_memory_utilization = 0.9

    args.max_num_requests = 1
    args.max_num_batched_tokens = 1024
    args.record_metrics = True
    args.frieren_executor_max_workers = 1
    args.scheduling = "sync"

    args.input_len = 1024 * 10
    args.hit_rate = 0.
    args.num_prompts = 10

    def test_vary_block_size(args):
        for block_size in [16, 32, 64, 128]:
            args.block_size = block_size
            process_warp_with_exc(benchmark, args)

    def test_prefill(args):
        args.output_len = 1
        test_vary_block_size(args)

    def test_decoding(args):
        args.output_len = 128
        test_vary_block_size(args)

    def test_vary_model(args):
        for model, quantization in [
            ("Qwen/Qwen2.5-7B-Instruct", None),
            ("Qwen/Qwen2.5-3B-Instruct", None),
            ("Qwen/Qwen2.5-7B-Instruct", "fp8"),
        ]:
            print(model, quantization)
            args.model = model
            args.tokenizer = args.model
            args.quantization = quantization

            test_prefill(args)
            test_decoding(args)

    test_vary_model(args)
