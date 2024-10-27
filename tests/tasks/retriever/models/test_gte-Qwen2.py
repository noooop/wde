import random
from typing import TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding, BatchFeature

from tests.tasks.utils import SentenceTransformersRunner, compare_embeddings

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


@pytest.fixture(scope="session")
def hf_runner():
    return SentenceTransformersRunner


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


MODELS = [
    "Alibaba-NLP/gte-Qwen2-1.5B-instruct", "Alibaba-NLP/gte-Qwen2-7B-instruct"
]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [3])
@pytest.mark.parametrize("scheduling", ["sync"])
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
    with hf_runner(model, dtype=dtype, auto_cls=AutoModel) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with wde_runner(model,
                    dtype=dtype,
                    max_num_requests=max_num_requests,
                    scheduling=scheduling,
                    switch_to_gte_Qwen2=True) as engine:
        outputs = engine.encode(example_prompts)

    similarities = compare_embeddings(hf_outputs, outputs)
    all_similarities = torch.stack(similarities)

    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
