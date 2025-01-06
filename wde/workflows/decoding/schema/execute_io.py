from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch

from wde.workflows.core.schema.execute_io import (ExecuteInput, ExecuteOutput,
                                                  ModelInput)

if TYPE_CHECKING:
    from concurrent.futures import Future

    from wde.workflows.core.backends.attention import AttentionMetadata
    from wde.workflows.decoding.backends.sampling.sampling_metadata import \
        SamplingMetadata
    from wde.workflows.decoding.backends.sampling.sampling_params import \
        SamplingType
    from wde.workflows.decoding.kv_cache.offloading.swap_out import SwapOutTask


@dataclass
class DecodingExecuteInput(ExecuteInput):
    swap_out_task: Optional["SwapOutTask"] = None


@dataclass
class DecodingModelInput(ModelInput):
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    attn_metadata: Optional["AttentionMetadata"] = None
    sampling_metadata: Optional[Union["SamplingMetadata", "Future"]] = None
    kv_caches: Optional[List[torch.Tensor]] = None

    def to(self, device, non_blocking=True):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=device,
                                                   non_blocking=non_blocking)
        return self

    def deferred_to(self, device, non_blocking=False):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "deferred_to"):
                continue
            self.__dict__[k] = self.__dict__[k].deferred_to(
                device=device, non_blocking=non_blocking)
        return self


SampleResultType = List[Tuple[List[int], List[int]]]
MultinomialSamplesType = Dict["SamplingType", torch.Tensor]
SampleResultsDictType = Dict[int, Tuple[List[int], List[int]]]


@dataclass
class SamplerOutput(ExecuteOutput):
    logprobs: Optional["torch.Tensor"] = None
    sampling_metadata: Optional["SamplingMetadata"] = None

    multinomial_samples: Optional[MultinomialSamplesType] = None
    greedy_samples: Optional[torch.Tensor] = None
    categorized_requests: Optional[Dict["SamplingType", List]] = None

    def to(self, device, non_blocking=True):
        for k in self.__dict__:

            if k == "logprobs":
                continue

            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=device,
                                                   non_blocking=non_blocking)

        for k in self.multinomial_samples:
            self.multinomial_samples[k] = self.multinomial_samples[k].to(
                device=device, non_blocking=non_blocking)

        return self


@dataclass
class Logprob:
    logprob: float
    rank: Optional[int] = None
    decoded_token: Optional[str] = None


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = List[Optional[Dict[int, Logprob]]]
# {token_id -> logprob} for each sequence group.
SampleLogprobs = List[Dict[int, Logprob]]


@dataclass
class Sample:
    output_token: int
    logprobs: Dict[int, Logprob]
