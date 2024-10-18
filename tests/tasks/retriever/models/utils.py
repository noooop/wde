from typing import List, TypeVar

import numpy as np
import torch
import torch.nn as nn
from transformers import BatchEncoding, BatchFeature

from tests.tasks.utils import cleanup, is_cpu

_T = TypeVar("_T", nn.Module, torch.Tensor, BatchEncoding, BatchFeature)


class FlagEmbeddingRunner:

    def wrap_device(self, input: _T) -> _T:
        if not is_cpu():
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

        self.model_name = model_name
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
