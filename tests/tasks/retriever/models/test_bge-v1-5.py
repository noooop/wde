import random

import numpy as np
import pytest
import torch

from tests.tasks.retriever.models.utils import FlagEmbeddingRunner
from tests.tasks.utils import compare_embeddings_np


@pytest.fixture(scope="session")
def hf_runner():
    return FlagEmbeddingRunner


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


MODELS = ['BAAI/bge-large-zh-v1.5', 'BAAI/bge-base-en-v1.5']


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_seqs", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["sync", "async", "double_buffer"])
@torch.inference_mode
def test_models(
    hf_runner,
    wde_runner,
    example_prompts,
    model: str,
    dtype: str,
    max_num_seqs: int,
    scheduling: str,
) -> None:
    with hf_runner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with wde_runner(model,
                    dtype=dtype,
                    max_num_seqs=max_num_seqs,
                    scheduling=scheduling) as engine:
        outputs = engine.encode(example_prompts)
        outputs = [t.cpu().numpy() for t in outputs]

    similarities = compare_embeddings_np(hf_outputs, outputs)
    all_similarities = np.stack(similarities)
    tolerance = 1e-2
    assert np.all((all_similarities <= 1.0 + tolerance)
                  & (all_similarities >= 1.0 - tolerance)
                  ), f"Not all values are within {tolerance} of 1.0"
