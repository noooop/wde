import random
from typing import List, TypeVar

import pytest
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BatchEncoding, BatchFeature

from tests.tasks.utils import HfRunner, compare_embeddings

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


@pytest.fixture(scope="session")
def hf_runner():
    return Qwen2HfRunner


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


MODELS = ["Qwen/Qwen2-0.5B-Instruct"]


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
    with hf_runner(model, dtype=dtype,
                   auto_cls=AutoModelForCausalLM) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with wde_runner(model,
                    dtype=dtype,
                    max_num_seqs=max_num_seqs,
                    scheduling=scheduling,
                    output_last_hidden_states=True) as engine:
        outputs = engine.encode(example_prompts)

    similarities = compare_embeddings(hf_outputs, outputs)
    all_similarities = torch.stack(similarities)
    tolerance = 1e-2
    assert torch.all((all_similarities <= 1.0 + tolerance)
                     & (all_similarities >= 1.0 - tolerance)
                     ), f"Not all values are within {tolerance} of 1.0"
