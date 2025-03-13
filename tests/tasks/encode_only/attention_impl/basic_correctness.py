import itertools as it
from typing import TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import BatchEncoding, BatchFeature

from tests.tasks.utils import (compare_embeddings, get_example_prompts,
                               wde_runner)
from wde.utils import process_warp
from wde.workflows.prefill_only.backends.attention.selector import \
    AttentionImpls

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)

MODELS = ["google-bert/bert-base-uncased"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
@pytest.mark.parametrize("max_num_requests", [2])
@pytest.mark.parametrize("scheduling", ["sync"])
@torch.inference_mode
def test_basic_correctness_fp16(
    model: str,
    dtype: str,
    max_num_requests: int,
    scheduling: str,
) -> None:
    example_prompts = get_example_prompts()
    attention_impls = AttentionImpls[dtype]

    impl_outputs_list = []

    for attention_impl in attention_impls:
        outputs = process_warp(wde_runner,
                               method="encode",
                               model=model,
                               example_prompts=example_prompts,
                               dtype=dtype,
                               max_num_requests=max_num_requests,
                               scheduling=scheduling,
                               attention_impl=attention_impl)

        impl_outputs_list.append((attention_impl, outputs))

    tolerance = 1e-2
    for a, b in it.combinations(impl_outputs_list, 2):
        similarities = compare_embeddings(a[1], b[1])
        all_similarities = torch.stack(similarities)

        assert torch.all(
            (all_similarities <= 1.0 + tolerance)
            & (all_similarities >= 1.0 - tolerance)
        ), f"{a[0]} vs {b[0]}, not all values are within {tolerance} of 1.0"
