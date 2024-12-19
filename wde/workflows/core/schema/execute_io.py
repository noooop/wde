from dataclasses import dataclass
from typing import Dict, Optional, Union

import msgspec
import torch


@dataclass
class ModelInput:

    def to(self, device, non_blocking=False):
        pass

    def deferred_to(self, device, non_blocking=False):
        pass


@dataclass
class WorkerInput:
    pass


@dataclass
class ExecuteInput:
    worker_input: Optional[WorkerInput]
    model_input: Optional[ModelInput]
    main_stream: Optional[torch.cuda.Stream] = None
    deferred_stream: Optional[torch.cuda.Stream] = None


@dataclass
class ExecuteOutput:
    execute_begin_ts: Optional[float] = None
    execute_end_ts: Optional[float] = None

    def to(self, target_device, non_blocking=False):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=target_device,
                                                   non_blocking=non_blocking)

    def deferred_to(self, target_device, non_blocking=False):
        pass


class IntermediateTensors(
        msgspec.Struct,
        omit_defaults=True,  # type: ignore[call-arg]
        array_like=True):  # type: ignore[call-arg]
    """For all pipeline stages except the last, we need to return the hidden
    states and residuals to be sent to the next stage. This data structure
    contains the hidden states and residuals for a request.
    """

    tensors: Dict[str, torch.Tensor]

    def __getitem__(self, key: Union[str, slice]):
        if isinstance(key, str):
            return self.tensors[key]
        elif isinstance(key, slice):
            return self.__class__({k: v[key] for k, v in self.tensors.items()})

    def __setitem__(self, key: str, value):
        self.tensors[key] = value

    def __len__(self):
        return len(self.tensors)

    def __eq__(self, other: object):
        return isinstance(other, self.__class__) and self

    def __repr__(self) -> str:
        return f"IntermediateTensors(tensors={self.tensors})"
