import random
import time


def benchmark(args):
    random.seed(args.seed)

    import sglang as sgl

    from wde.workflows.decoding.backends.sampling.utils import TokenSampler

    print(sgl.__version__)

    llm = sgl.Engine(model_path=args.model,
                     quantization=args.quantization,
                     dtype=args.dtype,
                     kv_cache_dtype=args.kv_cache_dtype,
                     device=args.device,
                     context_length=args.max_model_len,
                     max_running_requests=args.max_num_requests,
                     chunked_prefill_size=args.max_num_batched_tokens,
                     disable_radix_cache=not args.enable_prefix_caching)

    token_sampler = TokenSampler(args.tokenizer)

    prefix_len = int(args.hit_rate * args.input_len)
    unique_len = args.input_len - prefix_len
    prefix_token_ids = token_sampler.random_sample(prefix_len)

    prompts = []
    for _ in range(args.num_prompts):
        unique_part_token_ids = token_sampler.random_sample(unique_len)

        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        prompts.append(prompt_token_ids)

    sampling_params = dict(
        n=1,
        temperature=0.8,
        top_p=0.95,
        ignore_eos=True,
        max_new_tokens=args.output_len,
    )

    start = time.perf_counter()

    outputs = llm.generate(input_ids=prompts, sampling_params=sampling_params)

    for prompt, output in zip(prompts, outputs):
        pass

    end = time.perf_counter()
    elapsed_time = end - start

    print(f"Batchsize {args.max_num_requests}, Throughput: "
          f"{len(prompts) / elapsed_time:.4f} requests/s")

    llm.shutdown()


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.output_len = 16
    args.num_prompts = 1000

    args.seed = 0
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 1000

    args.enable_chunked_prefill = True
    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tensor_parallel_size = 1

    args.n = 1
    args.gpu_memory_utilization = 0.9
    args.hit_rate = 0.

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    def test_vary_max_num_batched_tokens(args):
        max_num_batched_tokens_list = [
            1536, 1024, 768, 512, 384, 256, 128, 64, 32
        ]

        for max_num_batched_tokens in max_num_batched_tokens_list:
            print("max_num_batched_tokens", max_num_batched_tokens)
            args.max_num_requests = max_num_batched_tokens
            args.max_num_batched_tokens = max_num_batched_tokens
            run(args)

    def test_vary_enable_prefix_caching(args):
        for enable_prefix_caching in [False, True]:
            args.enable_prefix_caching = enable_prefix_caching

            test_vary_max_num_batched_tokens(args)

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

            test_vary_enable_prefix_caching(args)

    test_vary_model(args)
