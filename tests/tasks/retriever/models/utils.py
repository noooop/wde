from typing import List, TypeVar

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchEncoding, BatchFeature
from vllm.platforms import current_platform

from tests.tasks.utils import cleanup
from wde.workflows.core.backends.models.transformers_utils.config import \
    model_overwrite

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class FlagEmbeddingRunner:

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
    ) -> None:
        # depend on FlagEmbedding peft
        from FlagEmbedding import FlagModel

        self.model_name = model_overwrite(model_name)

        model = FlagModel(self.model_name, use_fp16=dtype == "half")
        self.model = model

    @torch.inference_mode
    def encode(self, prompts: List[str]) -> np.ndarray:
        embeddings = self.model.encode(prompts)
        return embeddings

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


def hf_runner(model, dtype, example_prompts):
    with FlagEmbeddingRunner(model, dtype=dtype) as hf_model:
        hf_outputs = hf_model.encode(example_prompts)
    return hf_outputs
