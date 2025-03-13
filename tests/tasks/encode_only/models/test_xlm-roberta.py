from typing import TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import BatchEncoding, BatchFeature

from tests.tasks.utils import (compare_embeddings, get_example_prompts,
                               hf_runner, wde_runner)
from wde.utils import process_warp

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)

MODELS = ["FacebookAI/xlm-roberta-base", "FacebookAI/xlm-roberta-large"]


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
    example_prompts = get_example_prompts()

    hf_outputs = process_warp(hf_runner, model, dtype, example_prompts)
    outputs = process_warp(wde_runner,
                           method="encode",
                           model=model,
                           example_prompts=example_prompts,
                           dtype=dtype,
                           max_num_requests=max_num_requests,
                           scheduling=scheduling)

    similarities = compare_embeddings(hf_outputs, outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
