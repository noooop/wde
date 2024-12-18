import random
import time

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams, TokensPrompt

    from wde.workflows.decoding.backends.sampling.utils import TokenSampler

    print(vllm.__version__)

    engine_args = EngineArgs(
        model=args.model,
        quantization=args.quantization,
        tensor_parallel_size=args.tensor_parallel_size,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        disable_log_stats=True)

    engine = LLMEngine.from_engine_args(engine_args)
    token_sampler = TokenSampler(args.tokenizer)

    prefix_len = int(args.hit_rate * args.input_len)
    unique_len = args.input_len - prefix_len
    prefix_token_ids = token_sampler.random_sample(prefix_len)

    requests = []
    for _ in range(args.num_prompts):
        unique_part_token_ids = token_sampler.random_sample(unique_len)

        prompt_token_ids = prefix_token_ids + unique_part_token_ids
        requests.append(prompt_token_ids)

    start = time.perf_counter()
    for request_id, prompt_token_ids in enumerate(requests):
        inputs = TokensPrompt(prompt_token_ids=prompt_token_ids)
        sampling_params = SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=args.output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    timestamp = {}
    num_cached_tokens = {}

    n_step = 0
    while engine.has_unfinished_requests():
        n_step += 1
        request_outputs = engine.step()
        ts = time.perf_counter()

        for request in request_outputs:
            request_id = request.request_id
            if request_id not in timestamp:
                timestamp[request_id] = []
            timestamp[request_id].append(ts)

            if request_id not in num_cached_tokens:
                num_cached_tokens[request_id] = []

            num_cached_tokens[request_id].append(request.num_cached_tokens)

    end = time.perf_counter()

    tpot = []
    for v in timestamp.values():
        dd = [v[i] - v[i - 1] for i in range(1, len(v))]
        tpot.extend(dd)

    actual_hit_rate = np.mean([v[-1] for v in num_cached_tokens.values()
                               ]) / args.input_len
    tpot = np.mean(tpot)
    elapsed_time = end - start

    total_num_tokens = (args.input_len + args.output_len) * args.num_prompts

    print(
        f"Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
        f"{total_num_tokens / elapsed_time:.4f} tokens/s, "
        f"Latency {tpot*1000:0.2f} ms, n_step {n_step}, actual hit rate {actual_hit_rate}"
    )


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
    args.quantization_param_path = None
    args.tensor_parallel_size = 1

    args.n = 1
    args.enforce_eager = True
    args.enable_prefix_caching = False
    args.tokenizer = args.model
    args.gpu_memory_utilization = 0.9

    args.max_num_seqs = 32
    args.max_num_batched_tokens = 1024

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    args.enable_prefix_caching = False
    args.hit_rate = 0.
    run(args)

    args.enable_prefix_caching = True
    for hit_rate in [0.1 * x for x in range(0, 11)]:
        args.hit_rate = hit_rate
        run(args)
