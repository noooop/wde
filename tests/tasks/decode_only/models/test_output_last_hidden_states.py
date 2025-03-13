from typing import List, TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BatchEncoding, BatchFeature

from tests.tasks.utils import (HfRunner, compare_embeddings,
                               get_example_prompts, wde_runner)
from wde.utils import process_warp

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class Qwen2HfRunner(HfRunner):

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        outputs = self.model(**encoded_input, output_hidden_states=True)

        last_hidden_states = outputs.hidden_states[-1]
        seq_len = encoded_input.attention_mask.sum(axis=1)

        last_hidden_states_list = []
        for e, s in zip(last_hidden_states, seq_len):
            last_hidden_states_list.append(e[s - 1])
        return last_hidden_states_list


def hf_runner(model, dtype, example_prompts):
    with Qwen2HfRunner(model, dtype=dtype,
                       auto_cls=AutoModelForCausalLM) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    return hf_outputs


MODELS = ["Qwen/Qwen2.5-0.5B-Instruct"]


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
                           scheduling=scheduling,
                           output_last_hidden_states=True)

    similarities = compare_embeddings(hf_outputs, outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
