import random


def benchmark_wde(args):
    random.seed(args.seed)

    import torch

    from wde.tasks.encode_only.arg_utils import \
        EncodeOnlyEngineArgs as EngineArgs
    from wde.workflows.core.llm_engine import LLMEngine

    prompt = "if" * args.input_len
    requests = [prompt for _ in range(args.num_prompts)]

    engine_args = EngineArgs(
        model=args.model,
        tokenizer=args.tokenizer,
        quantization=args.quantization,
        seed=args.seed,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        quantization_param_path=args.quantization_param_path,
        device=args.device,
        max_num_requests=1,
        scheduling=args.scheduling,
        frieren_executor_max_workers=args.frieren_executor_max_workers,
    )

    engine = LLMEngine.from_engine_args(engine_args)

    for max_num_requests in args.batchsize:
        engine.engine_config.scheduler_config.set_args(
            max_num_requests=max_num_requests)

        for request_id, prompt in enumerate(requests):
            engine.add_request(str(request_id), prompt)

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
                                 f"{max_num_requests}.json")

        engine.scheduler.clear()
        engine.executor.shutdown_execute_loop()


if __name__ == '__main__':
    from easydict import EasyDict as edict
    args = edict()

    args.input_len = 32
    args.num_prompts = 10000

    args.model = 'BAAI/bge-m3'

    args.trust_remote_code = False
    args.tokenizer = args.model
    args.seed = 0
    args.quantization = None
    args.quantization_param_path = None
    args.max_model_len = None

    args.dtype = "half"
    args.device = "cuda"
    args.batchsize = [1, 2, 4, 8, 16, 32, 64]
    args.frieren_executor_max_workers = 1

    from concurrent.futures import ProcessPoolExecutor

    def run_wde(args):
        with ProcessPoolExecutor(1) as executor:
            f = executor.submit(benchmark_wde, args)
            f.result()

    for scheduling in ["sync", "simple_async"]:
        print(f"scheduling: {scheduling}")
        args.scheduling = scheduling
        run_wde(args)

    for scheduling in ["async"]:
        for max_workers in [1, 2, 3]:
            print(f"scheduling: {scheduling}-{max_workers}")
            args.frieren_executor_max_workers = max_workers
            args.scheduling = scheduling
            run_wde(args)
