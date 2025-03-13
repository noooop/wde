import pytest
import torch

from tests.engine.utils import WDEGeventRunner
from tests.tasks.utils import (bert_hf_runner, compare_embeddings,
                               get_example_prompts)
from wde.utils import process_warp

MODELS = ["google-bert/bert-base-uncased"]


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["async"])
@torch.inference_mode
def test_models(
    model: str,
    dtype: str,
    max_num_requests: int,
    scheduling: str,
) -> None:

    example_prompts = get_example_prompts()

    hf_outputs = process_warp(bert_hf_runner, model, dtype, example_prompts)

    with WDEGeventRunner(model,
                         dtype=dtype,
                         max_num_requests=max_num_requests,
                         scheduling=scheduling) as engine:
        outputs = engine.encode(example_prompts)

    similarities = compare_embeddings(hf_outputs, outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
