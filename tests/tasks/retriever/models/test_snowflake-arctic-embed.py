from typing import List, TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import AutoModel, BatchEncoding, BatchFeature

from tests.tasks.utils import (HfRunner, compare_embeddings,
                               get_example_prompts, wde_runner)
from wde.utils import process_warp

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class BertHfRunner(HfRunner):

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        embeddings = self.model(**encoded_input)[0][:, 0]
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


def hf_runner(model, dtype, example_prompts):
    with BertHfRunner(model,
                      dtype=dtype,
                      auto_cls=AutoModel,
                      model_kwargs={"add_pooling_layer": False}) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    return hf_outputs


MODELS = ['Snowflake/snowflake-arctic-embed-xs']


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [2])
@pytest.mark.parametrize("scheduling", ["sync"])
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
