from benchmarks.chat.util import get_requests, run
from wde.utils import process_warp_with_exc


def test(args, requests):
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs
    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        block_allocator=args.block_allocator,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_requests=args.max_num_requests,
        scheduling=args.scheduling,
        frieren_executor_max_workers=args.frieren_executor_max_workers,
        record_metrics=args.record_metrics)

    run(args, engine_args, requests)


def benchmark(args):
    requests = get_requests(args)
    process_warp_with_exc(test, args, requests)


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.output_len = 16
    args.num_prompts = 1000

    args.seed = 0
    args.dtype = 'auto'
    args.cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 1000

    args.quantization_param_path = None
    args.enable_prefix_caching = None
    args.gpu_memory_utilization = 0.9
    args.frieren_executor_max_workers = 1
    args.record_metrics = True

    args.hit_rate = 0.

    from wde.workflows.decoding.scheduler import BLOCK_ALLOCATOR_MAP

    def test_vary_max_num_batched_tokens(args):
        max_num_batched_tokens_list = [
            1536, 1024, 768, 512, 384, 256, 128, 64, 32
        ]

        for max_num_batched_tokens in max_num_batched_tokens_list:
            print("max_num_batched_tokens", max_num_batched_tokens)
            args.max_num_requests = max_num_batched_tokens
            args.max_num_batched_tokens = max_num_batched_tokens
            process_warp_with_exc(benchmark, args)

    def test_vary_scheduling(args):
        for scheduling in ["sync", "simple_async"]:
            print(f"scheduling: {scheduling}")
            args.scheduling = scheduling
            print()

            test_vary_max_num_batched_tokens(args)

        for scheduling in ["async"]:
            for max_workers in [1, 2, 3]:
                print(f"scheduling: {scheduling}-{max_workers}")
                args.frieren_executor_max_workers = max_workers
                args.scheduling = scheduling
                print()
                test_vary_max_num_batched_tokens(args)

    def test_vary_block_allocator(args):
        for block_allocator in list(BLOCK_ALLOCATOR_MAP.keys()):
            print("block_allocator", block_allocator)
            args.block_allocator = block_allocator

            test_vary_scheduling(args)

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
            test_vary_block_allocator(args)

    test_vary_model(args)
