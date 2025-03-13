import numpy as np
import pytest
import torch

from tests.tasks.utils import (bert_hf_runner, compare_embeddings_np,
                               get_example_prompts)
from wde.utils import process_warp

MODELS = ["google-bert/bert-base-uncased"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["async"])
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
    example_prompts = get_example_prompts()

    hf_outputs = process_warp(bert_hf_runner, model, dtype, example_prompts)
    hf_outputs = [x.cpu().numpy() for x in hf_outputs]

    with wde_runner(model,
                    dtype=dtype,
                    max_num_requests=max_num_requests,
                    scheduling=scheduling) as engine:
        outputs = engine.encode(example_prompts)

    similarities = compare_embeddings_np(hf_outputs, outputs)
    all_similarities = np.stack(similarities)
    tolerance = 1e-2
    assert np.all((all_similarities <= 1.0 + tolerance)
                  & (all_similarities >= 1.0 - tolerance)
                  ), f"Not all values are within {tolerance} of 1.0"
