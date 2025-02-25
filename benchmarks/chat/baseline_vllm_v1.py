import os

os.environ["VLLM_USE_V1"] = "1"

import time

from benchmarks.chat.util import get_requests
from wde.utils import process_warp_with_exc


def benchmark(args):

    import vllm
    from vllm import LLM, SamplingParams, TokensPrompt

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
              max_num_seqs=args.max_num_requests,
              disable_log_stats=True)

    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        pass

    end = time.perf_counter()

    elapsed_time = end - start

    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts

    print(f"Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
          f"{total_num_tokens / elapsed_time:.4f} tokens/s")

    del llm


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

    def test_vary_max_num_batched_tokens(args):
        max_num_batched_tokens_list = [
            1536, 1024, 768, 512, 384, 256, 128, 64, 32
        ]

        for max_num_batched_tokens in max_num_batched_tokens_list:
            print("max_num_batched_tokens", max_num_batched_tokens)
            args.max_num_requests = max_num_batched_tokens
            args.max_num_batched_tokens = max_num_batched_tokens
            process_warp_with_exc(benchmark, args)

    def test_vary_enable_prefix_caching(args):
        for enable_prefix_caching in [False, True]:
            args.enable_prefix_caching = enable_prefix_caching

            test_vary_max_num_batched_tokens(args)

    def test_vary_enforce_eager(args):
        for enforce_eager in [False, True]:
            args.enforce_eager = enforce_eager

            test_vary_enable_prefix_caching(args)

    def test_vary_model(args):
        for model, quantization in [
                # ("Qwen/Qwen2.5-7B-Instruct", None),
            ("Qwen/Qwen2.5-3B-Instruct", None),
            ("Qwen/Qwen2.5-7B-Instruct", "fp8"),
        ]:
            print(model, quantization)
            args.model = model
            args.tokenizer = args.model
            args.quantization = quantization

            test_vary_enforce_eager(args)

    test_vary_model(args)
