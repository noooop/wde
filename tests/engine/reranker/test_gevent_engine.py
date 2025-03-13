import numpy as np
import pytest
import torch

from tests.engine.utils import WDEGeventRunner
from tests.tasks.reranker.util import (get_reranker_example_prompts,
                                       hf_reranker_runner, sigmoid)
from wde.utils import process_warp

MODELS = ["BAAI/bge-reranker-v2-m3"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["sync", "async"])
@torch.inference_mode
def test_models(
    model: str,
    dtype: str,
    max_num_requests: int,
    scheduling: str,
) -> None:
    example_prompts = get_reranker_example_prompts()

    hf_outputs = process_warp(hf_reranker_runner, model, dtype,
                              example_prompts)

    with WDEGeventRunner(model, dtype=dtype,
                         max_num_requests=max_num_requests) as engine:
        outputs = engine.compute_score(example_prompts)

    # Without using sigmoid,
    # the difference may be greater than 1e-2, resulting in flakey test
    hf_outputs = [sigmoid(x) for x in hf_outputs]
    outputs = [sigmoid(x) for x in outputs]

    all_similarities = np.array(hf_outputs) - np.array(outputs)

    tolerance = 1e-2
    assert np.all((all_similarities <= tolerance)
                  & (all_similarities >= -tolerance)
                  ), f"Not all values are within {tolerance} of 1.0"
