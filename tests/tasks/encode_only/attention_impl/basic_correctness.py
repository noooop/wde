import itertools as it
import random
from typing import TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import BatchEncoding, BatchFeature

from tests.tasks.utils import compare_embeddings
from wde.tasks.prefill_only.backends.attention.selector import AttentionImpls

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


@pytest.fixture(scope="session")
def example_prompts():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 11
    random.shuffle(prompts)
    return prompts


MODELS = ["google-bert/bert-base-uncased"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["float", "half", "bfloat16"])
@pytest.mark.parametrize("max_num_seqs", [2])
@pytest.mark.parametrize("scheduling", ["sync"])
@torch.inference_mode
def test_basic_correctness_fp16(
    hf_runner,
    wde_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_seqs: int,
    scheduling: str,
) -> None:
    attention_impls = AttentionImpls[dtype]

    impl_outputs_list = []

    for attention_impl in attention_impls:
        with wde_runner(
                model,
                dtype=dtype,
                max_num_seqs=max_num_seqs,
                scheduling=scheduling,
                attention_impl=attention_impl,
        ) as engine:
            outputs = engine.encode(example_prompts)
            impl_outputs_list.append((attention_impl, outputs))

    tolerance = 1e-2
    for a, b in it.combinations(impl_outputs_list, 2):
        similarities = compare_embeddings(a[1], b[1])
        all_similarities = torch.stack(similarities)

        assert torch.all(
            (all_similarities <= 1.0 + tolerance)
            & (all_similarities >= 1.0 - tolerance)
        ), f"{a[0]} vs {b[0]}, not all values are within {tolerance} of 1.0"
