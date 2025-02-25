import time

from wde.utils import process_warp_with_exc


def benchmark(args):
    import os
    if args.use_v1:
        os.environ["VLLM_USE_V1"] = "1"
    else:
        os.environ["VLLM_USE_V1"] = "0"

    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

    from benchmarks.chat.util import get_requests

    print(vllm.__version__)

    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        ignore_eos=True,
        max_tokens=args.output_len,
    )

    requests = get_requests(args)

    prompts = []
    for prompt_token_ids in requests:
        inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
        prompts.append(inputs)

    llm = LLM(model=args.model,
              quantization=args.quantization,
              tensor_parallel_size=args.tensor_parallel_size,
              seed=args.seed,
              trust_remote_code=args.trust_remote_code,
              dtype=args.dtype,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              enforce_eager=args.enforce_eager,
              kv_cache_dtype=args.kv_cache_dtype,
              device=args.device,
              enable_prefix_caching=args.enable_prefix_caching,
              enable_chunked_prefill=args.enable_chunked_prefill,
              max_num_batched_tokens=args.max_num_batched_tokens,
              max_num_seqs=args.max_num_seqs,
              disable_log_stats=True)

    for n in range(args.get("repeat", 1)):
        start = time.perf_counter()
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            pass

        end = time.perf_counter()

        elapsed_time = end - start

        total_num_tokens = (args.input_len +
                            args.output_len) * args.num_prompts

        print(f"Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
              f"{total_num_tokens / elapsed_time:.4f} tokens/s")


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

    args.enable_chunked_prefill = True

    args.trust_remote_code = False
    args.tensor_parallel_size = 1

    args.n = 1
    args.enforce_eager = True
    args.enable_prefix_caching = False
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    args.max_num_seqs = 32
    args.max_num_batched_tokens = 1024

    def test_vary_hit_rate(args):
        for hit_rate in [0.1 * x for x in range(0, 11)]:
            args.hit_rate = hit_rate
            process_warp_with_exc(benchmark, args)

    def test_vary_enable_prefix_caching(args):
        for enable_prefix_caching in [False, True]:
            args.enable_prefix_caching = enable_prefix_caching

            test_vary_hit_rate(args)

    def test_vary_enforce_eager(args):
        for enforce_eager in [False, True]:
            args.enforce_eager = enforce_eager

            test_vary_enable_prefix_caching(args)

    def test_v0_v1(args):
        for use_v1 in [False, True]:
            args.use_v1 = use_v1
            test_vary_enforce_eager(args)

    test_v0_v1(args)
