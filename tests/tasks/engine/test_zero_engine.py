import os
import random
from typing import List, Optional, TypeVar

import numpy as np
import pytest
import torch
import torch.nn as nn
from gevent.pool import Pool
from transformers import BatchEncoding, BatchFeature, BertModel

from tests.tasks.utils import HfRunner, compare_embeddings_np

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


class WDEZERORunner:

    def __init__(self,
                 model_name: str,
                 max_num_seqs: int = 4,
                 tokenizer_name: Optional[str] = None,
                 dtype: str = "half",
                 scheduling: str = "async",
                 attention_impl: Optional[str] = None,
                 **kwargs) -> None:
        if attention_impl is not None:
            os.environ["WDE_ATTENTION_BACKEND"] = attention_impl
        torch.multiprocessing.set_start_method('spawn', force=True)
        from wde.engine.zero_engine import start_zero_engine

        self.model_name = model_name

        engine_args = {
            "model": model_name,
            "tokenizer": tokenizer_name,
            "dtype": dtype,
            "max_num_seqs": max_num_seqs,
            "scheduling": scheduling,
        }
        engine_args.update(**kwargs)

        self.handle = start_zero_engine(engine_args)

    def encode(self, prompts: List[str]) -> List[List[float]]:
        from wde.tasks.retriever.engine.client import RetrieverClient

        client = RetrieverClient()
        client.wait_service_available(self.model_name)

        def worker(prompt):
            output = client.encode(name=self.model_name, inputs=prompt)
            return output

        outputs = []
        p = Pool(2)
        for out in p.imap(worker, prompts):
            embedding = out.embedding
            outputs.append(embedding)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        for h in self.handle:
            h.terminate()


@pytest.fixture(scope="session")
def wde_runner():
    return WDEZERORunner


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
@pytest.mark.parametrize("max_num_seqs", [2, 3, 5, 7])
@pytest.mark.parametrize("scheduling", ["async", "double_buffer"])
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
    with hf_runner(model, dtype=dtype, auto_cls=BertModel) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)
        hf_outputs = [x.cpu().numpy() for x in hf_outputs]

    with wde_runner(model,
                    dtype=dtype,
                    max_num_seqs=max_num_seqs,
                    scheduling=scheduling) as engine:
        outputs = engine.encode(example_prompts)

    similarities = compare_embeddings_np(hf_outputs, outputs)
    all_similarities = np.stack(similarities)
    tolerance = 1e-2
    assert np.all((all_similarities <= 1.0 + tolerance)
                  & (all_similarities >= 1.0 - tolerance)
                  ), f"Not all values are within {tolerance} of 1.0"
