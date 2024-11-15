import torch

from wde import LLM, SamplingParams

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 4

sampling_params = SamplingParams(temperature=0.8, top_p=0.95)


def benchmark_vllm(args):
    llm = LLM(model="Qwen/Qwen2.5-1.5B-Instruct",
              quantization=None,
              max_model_len=128,
              scheduling=args.scheduling,
              max_num_seqs=2,
              max_num_batched_tokens=512)

    for i in range(2):
        llm.generate(prompts, sampling_params)

    with torch.profiler.profile(activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
    ]) as prof:
        llm.generate(prompts, sampling_params)

    prof.export_chrome_trace(f"{scheduling}_execute_loop.json")


if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor

    from easydict import EasyDict as edict
    args = edict()

    def run_vllm(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_vllm, args)
            f.result()

    for scheduling in ["sync", "simple_async", "async", "double_buffer"]:
        print(scheduling)
        args.scheduling = scheduling
        run_vllm(args)
