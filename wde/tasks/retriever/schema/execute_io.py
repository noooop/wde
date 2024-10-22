from dataclasses import dataclass

import torch

from wde.tasks.core.schema.execute_io import ExecuteOutput


@dataclass
class EmbeddingExecuteOutput(ExecuteOutput):
    embeddings: torch.Tensor
