import random
import time

import numpy as np


def benchmark(args):
    random.seed(args.seed)

    from wde import LLMEngine, SamplingParams
    from wde.workflows.decoding.arg_utils import \
        DecodingEngineArgs as EngineArgs

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        enable_prefix_caching=args.enable_prefix_caching,
        download_dir=args.download_dir,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        scheduling=args.scheduling)

    engine = LLMEngine.from_engine_args(engine_args)

    prompt = "hi" * (args.input_len - 1)
    requests = [(prompt, args.input_len, args.output_len)
                for _ in range(args.num_prompts)]

    start = time.perf_counter()
    metrics_list = []

    for request_id, (prompt, _, output_len) in enumerate(requests):
        inputs = prompt
        sampling_params = SamplingParams(
            n=args.n,
            temperature=1.0,
            top_p=1.0,
            ignore_eos=True,
            max_tokens=output_len,
        )
        engine.add_request(str(request_id), inputs, sampling_params)

    n_step = 0
    while engine.has_unfinished_requests():
        n_step += 1
        request_outputs = engine.step()
        for request in request_outputs:
            metrics_list.append(request.metrics)

    end = time.perf_counter()

    elapsed_time = end - start
    scheduling_time = np.mean([m.scheduling_time for m in metrics_list])
    num_requests = np.mean([m.num_requests for m in metrics_list])
    num_batched_tokens = np.mean([m.num_batched_tokens for m in metrics_list])

    scheduling2inference = np.mean(
        [m.scheduling2inference for m in metrics_list])
    inference_time = np.mean([m.inference_time for m in metrics_list])
    latency = np.mean([m.latency for m in metrics_list])
    avg_latency = elapsed_time / n_step

    print(
        f"Batchsize {args.max_num_seqs}, Throughput: "
        f"{len(requests) / elapsed_time:.4f} requests/s, "
        f"Scheduling time {scheduling_time * 1000:0.4f} ms, "
        f"Num requests {num_requests:.2f}, ",
        f"Num batched tokens {num_batched_tokens:.2f}, ",
        f"Scheduling2inference {scheduling2inference * 1000:0.4f} ms, "
        f"Inference time {inference_time * 1000:0.4f} ms, "
        f"Avg Latency {avg_latency * 1000:0.4f} ms, "
        f"Latency {latency * 1000:0.4f} ms, n_step {n_step}")


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.dataset = None
    args.input_len = 256
    args.output_len = 16

    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.trust_remote_code = False
    args.tokenizer = args.model
    args.quantization = "fp8"
    args.quantization_param_path = None
    args.seed = 0
    args.n = 1
    args.num_prompts = 1000
    args.dtype = 'auto'
    args.max_model_len = 1000

    args.kv_cache_dtype = "auto"
    args.device = "cuda"
    args.enable_prefix_caching = False
    args.gpu_memory_utilization = 0.9
    args.output_json = None
    args.download_dir = None

    import sys
    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    if "full" in sys.argv:
        max_num_seqs_list = [1536, 1024, 768, 512, 384, 256, 128, 64, 32]
    else:
        max_num_seqs_list = [256, 128]

    for scheduling in ["sync", "simple_async", "async", "double_buffer"]:
        print(f"scheduling: {scheduling}")
        args.scheduling = scheduling

        print()
        for max_num_seqs in max_num_seqs_list:
            print("max_num_seqs", max_num_seqs)
            args.max_num_seqs = max_num_seqs
            args.max_num_batched_tokens = args.max_num_seqs
            run(args)
