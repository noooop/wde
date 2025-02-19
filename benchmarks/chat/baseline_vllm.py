import random
import time

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    import vllm
    from vllm import EngineArgs, LLMEngine, SamplingParams, TextPrompt

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

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

    start = time.perf_counter()
    for request_id, (prompt, _, output_len) in enumerate(requests):
        inputs = TextPrompt(prompt=prompt)
        sampling_params = SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    n_step = 0
    timestamp = {}
    while engine.has_unfinished_requests():
        n_step += 1
        request_outputs = engine.step()
        ts = time.perf_counter()

        for request in request_outputs:
            request_id = request.request_id
            if request_id not in timestamp:
                timestamp[request_id] = []
            timestamp[request_id].append(ts)

    end = time.perf_counter()

    tpot = []
    for v in timestamp.values():
        dd = [v[i] - v[i - 1] for i in range(1, len(v))]
        tpot.extend(dd)

    tpot = np.mean(tpot)
    elapsed_time = end - start

    total_num_tokens = sum(prompt_len + output_len
                           for _, prompt_len, output_len in requests)

    print(f"Throughput: {len(requests) / elapsed_time:.4f} requests/s, "
          f"{total_num_tokens / elapsed_time:.4f} tokens/s, "
          f"Delay {tpot*1000:0.2f} ms, n_step {n_step}")


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 256
    args.output_len = 16
    args.num_prompts = 1000

    args.seed = 0
    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.quantization = "fp8"
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 1000

    args.enable_chunked_prefill = True

    args.trust_remote_code = False
    args.quantization_param_path = None
    args.tensor_parallel_size = 1

    args.n = 1

    args.enable_prefix_caching = False
    args.gpu_memory_utilization = 0.9

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    for enforce_eager in [False, True]:
        args.enforce_eager = enforce_eager

        max_num_batched_tokens_list = [
            1536, 1024, 768, 512, 384, 256, 128, 64, 32
        ]

        for max_num_batched_tokens in max_num_batched_tokens_list:
            print("max_num_batched_tokens", max_num_batched_tokens)
            args.max_num_seqs = max_num_batched_tokens
            args.max_num_batched_tokens = max_num_batched_tokens
            run(args)
