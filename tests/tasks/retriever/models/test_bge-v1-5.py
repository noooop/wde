import numpy as np
import pytest
import torch

from tests.tasks.retriever.models.utils import hf_runner
from tests.tasks.utils import (compare_embeddings_np, get_example_prompts,
                               wde_runner)
from wde.utils import process_warp

MODELS = ['BAAI/bge-large-zh-v1.5', 'BAAI/bge-base-en-v1.5']


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
    outputs = [t.cpu().numpy() for t in outputs]

    similarities = compare_embeddings_np(hf_outputs, outputs)
    all_similarities = np.stack(similarities)
    tolerance = 1e-2
    assert np.all((all_similarities <= 1.0 + tolerance)
                  & (all_similarities >= 1.0 - tolerance)
                  ), f"Not all values are within {tolerance} of 1.0"
