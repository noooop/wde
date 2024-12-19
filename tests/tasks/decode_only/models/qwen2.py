import time

import pytest

from tests.tasks.decode_only.util import (HfDecodingRunner, WDERunner,
                                          check_logprobs_close)

MODELS = ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"]


@pytest.fixture(scope="session")
def wde_runner():
    return WDERunner


@pytest.fixture(scope="session")
def hf_runner():
    return HfDecodingRunner


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["bfloat16"])
@pytest.mark.parametrize("max_tokens", [5])
@pytest.mark.parametrize("scheduling", ["sync", "simple_async", "async"])
@pytest.mark.parametrize("enable_prefix_caching", [False, True])
def test_models(hf_runner, wde_runner, example_prompts, model: str, dtype: str,
                max_tokens: int, scheduling: str,
                enable_prefix_caching: bool) -> None:

    NUM_LOG_PROBS = 4

    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.generate_greedy_logprobs(example_prompts,
                                                       max_tokens,
                                                       NUM_LOG_PROBS)
    with wde_runner(model,
                    dtype=dtype,
                    scheduling=scheduling,
                    enable_prefix_caching=enable_prefix_caching) as wde_model:
        outputs = wde_model.generate_greedy_logprobs(example_prompts,
                                                     max_tokens, NUM_LOG_PROBS)

    check_logprobs_close(
        outputs_0_lst=hf_outputs,
        outputs_1_lst=outputs,
        name_0="hf",
        name_1="wde",
    )

    time.sleep(1)
