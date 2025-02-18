import random
import time

import pytest
from easydict import EasyDict as edict

from benchmarks.remote_kv_cache.util import (kv_cache_info,
                                             process_warp_with_exc,
                                             start_remote_kv_cache,
                                             wait_service_available)
from tests.tasks.decode_only.util import (HfDecodingRunner, WDERunner,
                                          check_logprobs_close)

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


@pytest.fixture(scope="session")
def example_prompts():
    _prompts = [
        "JDK is developed by",
        "Birds can",
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    prompts = []
    for i in range(1, 32):
        prompts.extend([p * i for p in _prompts])

    random.shuffle(prompts)
    return prompts


def wde_runner(args, example_prompts, NUM_LOG_PROBS):
    with WDERunner(
            args.model,
            dtype=args.dtype,
            scheduling=args.scheduling,
            enable_prefix_caching=args.enable_prefix_caching,
            swap_space=args.swap_space,
            remote_kv_cache_server=args.remote_kv_cache_server) as wde_model:
        outputs = wde_model.generate_greedy_logprobs(example_prompts,
                                                     args.max_tokens,
                                                     NUM_LOG_PROBS)

    return outputs


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("scheduling", ["sync", "simple_async", "async"])
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_remote_kv_cache(example_prompts, model: str, dtype: str,
                         max_tokens: int, scheduling: str,
                         enable_prefix_caching: bool) -> None:

    example_prompts = example_prompts * 4

    args = edict()

    args.seed = 0
    args.model = model
    args.block_size = 16
    args.swap_space = 40
    args.memory_space = 40
    args.dtype = dtype
    args.cache_dtype = "auto"
    args.scheduling = scheduling
    args.max_tokens = max_tokens
    args.enable_prefix_caching = enable_prefix_caching
    args.remote_kv_cache_server = True
    args.remote_kv_cache_server_name = f"kv_cache:{args.model}:{args.block_size}"

    NUM_LOG_PROBS = 4

    with HfDecodingRunner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       NUM_LOG_PROBS)

    server = start_remote_kv_cache(args)
    process_warp_with_exc(wait_service_available, args)
    process_warp_with_exc(kv_cache_info, args)

    outputs = process_warp_with_exc(wde_runner, args, example_prompts,
                                    NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=outputs,
        name_0="hf",
        name_1="wde",
    )

    process_warp_with_exc(kv_cache_info, args)

    outputs = process_warp_with_exc(wde_runner, args, example_prompts,
                                    NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=outputs,
        name_0="hf",
        name_1="wde",
    )

    process_warp_with_exc(kv_cache_info, args)

    for s in server:
        s.terminate()

    time.sleep(1)
