from dataclasses import dataclass

import torch

from wde.workflows.core.backends.attention import AttentionMetadata
from wde.workflows.core.schema.execute_io import ExecuteInput, ModelInput


@dataclass
class ModelInputForGPU(ModelInput):
    input_ids: torch.Tensor
    positions: torch.Tensor
    attn_metadata: AttentionMetadata

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)

    def to_dict(self):
        out = self.__dict__

        if "kv_caches" not in out:
            out["kv_caches"] = None

        return out


class PrefillOnlyExecuteInput(ExecuteInput):
    worker_input = None
    model_input: ModelInputForGPU
