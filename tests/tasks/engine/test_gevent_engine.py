import os
import random
from typing import List, Optional, TypeVar

import pytest
import shortuuid
import torch
import torch.nn as nn
from gevent.pool import Pool
from transformers import BatchEncoding, BatchFeature, BertModel

from tests.tasks.utils import HfRunner, cleanup, compare_embeddings
from wde.engine.gevent_engine import GeventLLMEngine

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class BertHfRunner(HfRunner):

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        outputs = self.model(**encoded_input).pooler_output
        return outputs


class WDEGeventRunner:

    def __init__(self,
                 model_name: str,
                 max_num_requests: int = 4,
                 tokenizer_name: Optional[str] = None,
                 dtype: str = "half",
                 scheduling: str = "async",
                 attention_impl: Optional[str] = None,
                 **kwargs) -> None:
        if attention_impl is not None:
            os.environ["WDE_ATTENTION_BACKEND"] = attention_impl

        self.model = GeventLLMEngine(model=model_name,
                                     tokenizer=tokenizer_name,
                                     trust_remote_code=True,
                                     max_num_requests=max_num_requests,
                                     dtype=dtype,
                                     scheduling=scheduling,
                                     **kwargs)

        if attention_impl is not None:
            assert (self.model.llm_engine.attn_backend.get_name().lower() ==
                    attention_impl.lower())

    def encode(self, prompts: List[str]) -> List[List[float]]:

        def worker(prompt):
            request_id = f"{shortuuid.random(length=22)}"
            outputs = self.model.encode(inputs=prompt, request_id=request_id)
            return list(outputs)[0]

        outputs = []
        p = Pool(2)
        for out in p.imap(worker, prompts):
            embedding = out.outputs
            outputs.append(embedding)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


@pytest.fixture(scope="session")
def wde_runner():
    return WDEGeventRunner


@pytest.fixture(scope="session")
def hf_runner():
    return BertHfRunner


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
@pytest.mark.parametrize("dtype", ["half"])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["async", "double_buffer"])
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
    with hf_runner(model, dtype=dtype, auto_cls=BertModel) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)

    with wde_runner(model,
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
