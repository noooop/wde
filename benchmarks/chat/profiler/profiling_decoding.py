def benchmark(args):
    import torch

    import wde
    from wde import LLMEngine, SamplingParams
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs

    print(wde.__version__)

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        enable_prefix_caching=args.enable_prefix_caching,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_requests=args.max_num_requests,
        scheduling=args.scheduling,
        frieren_executor_max_workers=args.frieren_executor_max_workers,
        record_metrics=args.record_metrics)

    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

    for request_id, (prompt, _, output_len) in enumerate(requests):
        inputs = prompt
        sampling_params = SamplingParams(
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    for i in range(20):
        engine.step()

    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        for i in range(10):
            engine.step()
    prof.export_chrome_trace(f"{scheduling}-"
                             f"{args.frieren_executor_max_workers}-"
                             f"{args.max_num_requests}.json")


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.output_len = 16
    args.num_prompts = 1000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-3B-Instruct"
    args.quantization = None
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 1000

    args.tokenizer = args.model
    args.quantization_param_path = None
    args.enable_prefix_caching = False
    args.gpu_memory_utilization = 0.9
    args.frieren_executor_max_workers = 1
    args.record_metrics = True

    from concurrent.futures import ProcessPoolExecutor

    def run_wde(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    max_num_batched_tokens_list = [1536, 1024, 768, 512, 384, 256, 128, 64, 32]

    for scheduling in ["sync", "simple_async"]:
        print(f"scheduling: {scheduling}")
        args.scheduling = scheduling
        print()
        for max_num_batched_tokens in max_num_batched_tokens_list:
            print("max_num_batched_tokens", max_num_batched_tokens)
            args.max_num_requests = max_num_batched_tokens
            args.max_num_batched_tokens = max_num_batched_tokens
            run_wde(args)

    for scheduling in ["async"]:
        for max_workers in [1, 2, 3]:
            print(f"scheduling: {scheduling}-{max_workers}")
            args.frieren_executor_max_workers = max_workers
            args.scheduling = scheduling

            print()
            for max_num_batched_tokens in max_num_batched_tokens_list:
                print("max_num_batched_tokens", max_num_batched_tokens)
                args.max_num_requests = max_num_batched_tokens
                args.max_num_batched_tokens = max_num_batched_tokens
                run_wde(args)
