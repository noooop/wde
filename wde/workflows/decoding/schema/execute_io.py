from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from wde.workflows.core.schema.execute_io import (ExecuteInput, ExecuteOutput,
                                                  ModelInput, WorkerInput)
from wde.workflows.decoding.backends.sampling_metadata import (
    SamplingMetadata, SequenceGroupToSample)
from wde.workflows.decoding.backends.sampling_params import SamplingType
from wde.workflows.decoding.backends.sequence import Logprob, PromptLogprobs


@dataclass
class DecodingModelInputForGPU(ModelInput):
    """
    This base class contains metadata needed for the base model forward pass
    but not metadata for possible additional steps, e.g., sampling. Model
    runners that run additional steps should subclass this method to add
    additional fields.
    """
    input_tokens: Optional[torch.Tensor] = None
    input_positions: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    query_lens: Optional[List[int]] = None
    attn_metadata: Optional["AttentionMetadata"] = None

    def to(self, device, non_blocking=True):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=device,
                                                   non_blocking=non_blocking)
        return self


@dataclass
class DecodingModelInputForGPUWithSamplingMetadata(DecodingModelInputForGPU):
    """
    Used by the ModelRunner.
    """
    sampling_metadata: Optional["SamplingMetadata"] = None
    # Used for speculative decoding. We do not broadcast it because it is only
    # used by the driver worker.
    is_prompt: Optional[bool] = None


@dataclass
class DecodingWorkerInputForGPU(WorkerInput):
    num_requests: int = 0
    blocks_to_swap_in: Optional[torch.Tensor] = None
    blocks_to_swap_out: Optional[torch.Tensor] = None
    blocks_to_copy: Optional[torch.Tensor] = None


@dataclass
class DecodingExecuteInput(ExecuteInput):
    worker_input: Optional[DecodingWorkerInputForGPU]
    model_input: Optional[DecodingModelInputForGPUWithSamplingMetadata]


class SequenceOutput:
    """The model output associated with a sequence.

    Args:
        parent_seq_id: The ID of the parent sequence (for forking in beam
            search).
        output_token: The output token ID.
        logprobs: The logprobs of the output token.
            (Token id -> logP(x_i+1 | x_0, ..., x_i))
    """

    def __init__(
        self,
        parent_seq_id: int,
        output_token: int,
        logprobs: Dict[int, Logprob],
    ) -> None:
        self.parent_seq_id = parent_seq_id
        self.output_token = output_token
        self.logprobs = logprobs

    def __repr__(self) -> str:
        return (f"SequenceOutput(parent_seq_id={self.parent_seq_id}, "
                f"output_token={self.output_token}, "
                f"logprobs={self.logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SequenceOutput):
            raise NotImplementedError()
        equal = (self.parent_seq_id == other.parent_seq_id
                 and self.output_token == other.output_token)
        log_probs_equal = other.logprobs == self.logprobs
        return equal and log_probs_equal


class SequenceGroupOutput(ABC):
    """The base class for model outputs associated with a sequence group."""

    @abstractmethod
    def __repr__(self) -> str:
        pass

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        pass


class CompletionSequenceGroupOutput(SequenceGroupOutput):
    """The model output associated with a completion sequence group."""

    def __init__(
        self,
        samples: List[SequenceOutput],
        prompt_logprobs: Optional[PromptLogprobs],
    ) -> None:
        self.samples = samples
        # Prompt logprob for each prompt query token.
        self.prompt_logprobs = prompt_logprobs

    def __repr__(self) -> str:
        return (f"CompletionSequenceGroupOutput(samples={self.samples}, "
                f"prompt_logprobs={self.prompt_logprobs})")

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CompletionSequenceGroupOutput):
            raise NotImplementedError()
        return (self.samples == other.samples
                and self.prompt_logprobs == other.prompt_logprobs)


# (num_token_ids, num_parent_ids) per sequence group.
SampleResultType = List[Tuple[List[int], List[int]]]

# Types of temporary data structures used for
# computing sample_result
SampleMetadataType = Dict[SamplingType, Tuple[List[int],
                                              List[SequenceGroupToSample]]]
MultinomialSamplesType = Dict[SamplingType, torch.Tensor]
SampleResultsDictType = Dict[int, Tuple[List[int], List[int]]]


@dataclass
class SampleResultArgsType:
    sample_metadata: SampleMetadataType
    multinomial_samples: MultinomialSamplesType
    sample_results_dict: SampleResultsDictType
    sampling_metadata: SamplingMetadata
    greedy_samples: Optional[torch.Tensor]


# Union of non-deferred (single-step scheduling)
# vs deferred (multi-step scheduling)
# sample result types
MaybeDeferredSampleResultType = Union[SampleResultType, SampleResultArgsType]

# Abbreviation of the _sample() return type
SampleReturnType = Tuple[MaybeDeferredSampleResultType, Optional[torch.Tensor]]


@dataclass
class SamplerOutput(ExecuteOutput):
    logprobs: Optional["torch.Tensor"] = None

    sample_metadata: Optional[SampleMetadataType] = None
    multinomial_samples: Optional[MultinomialSamplesType] = None
    sample_results_dict: Optional[SampleResultsDictType] = None
    sampling_metadata: Optional[SamplingMetadata] = None
    greedy_samples: Optional[torch.Tensor] = None

    def to(self, target_device, non_blocking=True):
        for k in self.multinomial_samples:
            self.multinomial_samples[k] = self.multinomial_samples[k].to(
                device=target_device, non_blocking=non_blocking)

        if isinstance(self.greedy_samples, torch.Tensor):
            self.greedy_samples = self.greedy_samples.to(
                device=target_device, non_blocking=non_blocking)
        return self
