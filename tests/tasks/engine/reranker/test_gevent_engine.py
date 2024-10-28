import random
from typing import TypeVar

import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import (AutoModelForSequenceClassification, BatchEncoding,
                          BatchFeature)

from tests.tasks.engine.utils import WDEGeventRunner
from tests.tasks.utils import HfRerankerRunner, sigmoid

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


@pytest.fixture(scope="session")
def hf_runner():
    return HfRerankerRunner


@pytest.fixture(scope="session")
def wde_runner():
    return WDEGeventRunner


@pytest.fixture(scope="session")
def example_prompts():
    pairs = [
        ["query", "passage"],
        ["what is panda?", "hi"],
        [
            "what is panda?",
            "The giant panda (Ailuropoda melanoleuca), "
            "sometimes called a panda bear or simply panda, "
            "is a bear species endemic to China.",
        ],
    ] * 11
    random.shuffle(pairs)
    return pairs


MODELS = ["BAAI/bge-reranker-v2-m3"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["sync", "async", "double_buffer"])
@torch.inference_mode
def test_models(
    hf_runner,
    wde_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_requests: int,
    scheduling: str,
) -> None:
    with hf_runner(model,
                   dtype=dtype,
                   auto_cls=AutoModelForSequenceClassification) as hf_model:
        hf_outputs = hf_model.compute_score(example_prompts)

    with wde_runner(model, dtype=dtype,
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
