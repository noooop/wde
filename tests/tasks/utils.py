import gc
import os
import random
from typing import Any, Dict, List, Optional, TypeVar

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer, BatchEncoding,
                          BatchFeature)
from vllm.platforms import current_platform
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE

from wde import LLM
from wde.tasks.reranker.schema.engine_io import RerankerInputs
from wde.workflows.core.backends.models.transformers_utils.config import \
    maybe_model_redirect

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class WDERunner:

    def __init__(self,
                 model_name: str,
                 max_num_requests: int = 4,
                 tokenizer_name: Optional[str] = None,
                 dtype: str = "half",
                 scheduling: str = "sync",
                 attention_impl: Optional[str] = None,
                 **kwargs) -> None:
        if attention_impl is not None:
            os.environ["WDE_ATTENTION_BACKEND"] = attention_impl

        self.model = LLM(model=model_name,
                         tokenizer=tokenizer_name,
                         trust_remote_code=True,
                         max_num_requests=max_num_requests,
                         dtype=dtype,
                         scheduling=scheduling,
                         **kwargs)

        if attention_impl is not None:
            assert (self.model.engine.attn_backend.get_name().lower() ==
                    attention_impl.lower())

    def encode(self, prompts: List[str]) -> List[List[float]]:
        req_outputs = self.model.encode(prompts)
        outputs = []
        for req_output in req_outputs:
            embedding = req_output.outputs
            outputs.append(embedding)
        return outputs

    def compute_score(self, inputs: RerankerInputs) -> List[float]:
        req_outputs = self.model.compute_score(inputs)
        outputs = []
        for req_output in req_outputs:
            score = req_output.score
            outputs.append(score)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class HfRunner:

    def wrap_device(self, input: _T) -> _T:
        if not current_platform.is_cpu():
            # Check if the input is already on the GPU
            if hasattr(input, "device") and input.device.type == "cuda":
                return input  # Already on GPU, no need to move
            return input.to("cuda")
        else:
            # Check if the input is already on the CPU
            if hasattr(input, "device") and input.device.type == "cpu":
                return input  # Already on CPU, no need to move
            return input.to("cpu")

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        auto_cls=AutoModelForCausalLM,
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = maybe_model_redirect(model_name)

        model_kwargs = model_kwargs if model_kwargs is not None else {}

        self.model = self.wrap_device(
            auto_cls.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
                **model_kwargs,
            ))

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        last_hidden_states = self.model(
            **encoded_input, output_hidden_states=True).hidden_states
        return last_hidden_states[-1][:, 0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class SentenceTransformersRunner(HfRunner):

    def __init__(
        self,
        model_name: str,
        dtype: str = "half",
        *,
        model_kwargs: Optional[Dict[str, Any]] = None,
        auto_cls=AutoModelForCausalLM,
    ) -> None:
        torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[dtype]

        self.model_name = maybe_model_redirect(model_name)

        from sentence_transformers import SentenceTransformer
        self.model = self.wrap_device(
            SentenceTransformer(model_name,
                                device="cpu",
                                trust_remote_code=True).to(dtype=torch_dtype))

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> torch.Tensor:
        return self.model.encode(prompts,
                                 convert_to_numpy=False,
                                 normalize_embeddings=False)


class HfRerankerRunner(HfRunner):

    @torch.inference_mode
    def compute_score(self, inputs: RerankerInputs) -> List[float]:
        encoded_input = self.tokenizer(inputs,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        scores = self.model(**encoded_input).logits.view(-1, )
        return scores.cpu().numpy().tolist()


class BertHfRunner(HfRunner):

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> List[List[torch.Tensor]]:
        encoded_input = self.tokenizer(prompts,
                                       padding=True,
                                       truncation=True,
                                       return_tensors="pt").to("cuda")

        outputs = self.model(**encoded_input).pooler_output
        return outputs


def compare_embeddings(embeddings1, embeddings2):

    def to_cuda_tenser(e):
        if isinstance(e, np.ndarray):
            e = torch.tensor(e)

        assert isinstance(e, torch.Tensor)

        return e.cuda()

    similarities = [
        F.cosine_similarity(to_cuda_tenser(e1), to_cuda_tenser(e2), dim=0)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


def compare_embeddings_np(embeddings1, embeddings2):

    def cosine_similarity_np(e1, e2):
        return e1.dot(e2) / (np.linalg.norm(e1) * np.linalg.norm(e2))

    similarities = [
        cosine_similarity_np(e1, e2)
        for e1, e2 in zip(embeddings1, embeddings2)
    ]
    return similarities


def cleanup():
    gc.collect()
    if not current_platform.is_cpu():
        torch.cuda.empty_cache()


def hf_runner(model, dtype, example_prompts):
    with HfRunner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)
    return hf_outputs


def bert_hf_runner(model, dtype, example_prompts):
    from transformers import BertModel
    with BertHfRunner(model, dtype=dtype, auto_cls=BertModel) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)
        return hf_outputs


def st_runner(model, dtype, example_prompts):
    with SentenceTransformersRunner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)
    return hf_outputs


def wde_runner(model, method, example_prompts, *args, **kwargs):
    with WDERunner(model, *args, **kwargs) as engine:
        outputs = getattr(engine, method)(example_prompts)
    return outputs


def get_example_prompts():
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 11
    random.shuffle(prompts)
    return prompts
