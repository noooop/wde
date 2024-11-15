from dataclasses import dataclass
from typing import List, Optional

import torch

from wde.workflows.core.schema.engine_io import RequestOutput


@dataclass
class EmbeddingRequestOutput(RequestOutput):
    prompt_token_ids: Optional[List[int]] = None
    outputs: Optional[torch.Tensor] = None
