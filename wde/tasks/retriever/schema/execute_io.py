from dataclasses import dataclass
from typing import Optional

import torch

from wde.workflows.core.schema.execute_io import ExecuteOutput


@dataclass
class RetrieverExecuteOutput(ExecuteOutput):
    embeddings: Optional[torch.Tensor] = None
