import random
import time


def benchmark(args):
    random.seed(args.seed)

    import sglang as sgl

    print(sgl.__version__)

    llm = sgl.Engine(
        model_path=args.model,
        quantization=args.quantization,
        dtype=args.dtype,
        kv_cache_dtype=args.kv_cache_dtype,
        device=args.device,
        context_length=args.max_model_len,
        # reverse Automatically adjust --chunked-prefill-size for small GPUs.
        chunked_prefill_size=args.max_num_batched_tokens * 4,
        disable_radix_cache=True,
        disable_disk_cache=True)

    sampling_params = dict(
        n=1,
        temperature=1.0,
        top_p=1.0,
        ignore_eos=True,
        max_new_tokens=args.output_len,
    )

    prompt = "hi" * (args.input_len - 1)
    prompts = [prompt for _ in range(args.num_prompts)]

    start = time.perf_counter()

    outputs = llm.generate(prompts, sampling_params)

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
    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.quantization = "fp8"
    args.dtype = 'auto'
    args.kv_cache_dtype = "auto"
    args.device = "cuda"

    args.max_model_len = 1000

    from concurrent.futures import ProcessPoolExecutor

    def run(args):
        try:
            with ProcessPoolExecutor(1) as executor:
                f = executor.submit(benchmark, args)
                f.result()
        except Exception:
            import traceback
            traceback.print_exc()

    max_num_batched_tokens_list = [1536, 1024, 768, 512, 384, 256, 128, 64, 32]

    for max_num_batched_tokens in max_num_batched_tokens_list:
        print("max_num_batched_tokens", max_num_batched_tokens)
        args.max_num_requests = max_num_batched_tokens
        args.max_num_batched_tokens = max_num_batched_tokens
        run(args)