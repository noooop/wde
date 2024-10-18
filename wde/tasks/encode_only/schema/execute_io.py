from dataclasses import dataclass
from typing import Optional

import torch

from wde.tasks.core.schema.execute_io import ExecuteOutput


@dataclass
class EncodeOnlyExecuteOutput(ExecuteOutput):
    last_hidden_states: torch.Tensor
    pooled_output: Optional[torch.Tensor] = None
