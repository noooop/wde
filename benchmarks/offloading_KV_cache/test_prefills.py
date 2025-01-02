import random
import time


def benchmark(args):
    random.seed(args.seed)

    import wde
    from wde import LLMEngine, SamplingParams
    from wde.workflows.core.schema.engine_io import TokensPrompt
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs
    from wde.workflows.decoding.backends.sampling.utils import TokenSampler

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
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_requests=args.max_num_requests,
        scheduling=args.scheduling,
        frieren_executor_max_workers=args.frieren_executor_max_workers,
        record_metrics=args.record_metrics,
        kv_cache_manager=args.kv_cache_manager,
        trust_remote_code=args.trust_remote_code)

    engine = LLMEngine.from_engine_args(engine_args)
    token_sampler = TokenSampler(args.tokenizer,
                                 trust_remote_code=args.trust_remote_code)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    for request_id in range(args.num_prompts):
        prompt_token_ids = token_sampler.random_sample(args.input_len)

        inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
        engine.add_request(str(request_id), inputs, sampling_params)

        start = time.perf_counter()
        n_step = 0
        while engine.has_unfinished_requests():
            n_step += 1
            engine.step()

        end = time.perf_counter()

        elapsed_time = end - start

        print(request_id, elapsed_time, n_step)


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    from easydict import EasyDict as edict

    args = edict()

    args.input_len = 8192
    args.output_len = 1
    args.num_prompts = 3

    args.seed = 0
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 10000
    args.trust_remote_code = True
    args.quantization_param_path = None
    args.gpu_memory_utilization = 0.9

    args.max_num_requests = 4
    args.max_num_batched_tokens = 1024
    args.record_metrics = True
    args.frieren_executor_max_workers = 1
    args.kv_cache_manager = "naive"
    args.scheduling = "sync"

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    items = [
        ("Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4", None),
        ("Qwen/Qwen2.5-7B-Instruct", None),
        ("Qwen/Qwen2.5-7B-Instruct", "fp8"),
        ("Qwen/Qwen2.5-3B-Instruct", None),
        ("THUDM/glm-4-9b-chat-1m", None),
        ("THUDM/glm-4-9b-chat-1m", "fp8"),
        ("NousResearch/Hermes-3-Llama-3.1-8B", None),
        ("NousResearch/Hermes-3-Llama-3.1-8B", "fp8"),
    ]

    for model, quantization in items:
        print(model, quantization)
        args.model = model
        args.tokenizer = args.model
        args.quantization = quantization
        run(args)
