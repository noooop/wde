import enum
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch

from wde.workflows.core.schema.engine_io import SchedulableRequest

if TYPE_CHECKING:
    from vllm.core.block.block_table import BlockTable

    from wde.workflows.decoding.backends.sampling.sampling_params import \
        SamplingParams
    from wde.workflows.decoding.schema.execute_io import (Logprob,
                                                          PromptLogprobs,
                                                          SampleLogprobs)


class RequestStatus(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    SWAPPED = 2
    # Note: anything after SWAPPED (2) will be considered
    # as a finished status.
    FINISHED_STOPPED = 3
    FINISHED_LENGTH_CAPPED = 4
    FINISHED_ABORTED = 5
    FINISHED_IGNORED = 6

    @staticmethod
    def is_finished(status: "RequestStatus") -> bool:
        return status > RequestStatus.SWAPPED

    @staticmethod
    def get_finished_reason(status: "RequestStatus") -> Union[str, None]:
        if status == RequestStatus.FINISHED_STOPPED:
            finish_reason = "stop"
        elif status == RequestStatus.FINISHED_LENGTH_CAPPED:
            finish_reason = "length"
        elif status == RequestStatus.FINISHED_ABORTED:
            finish_reason = "abort"
        elif status == RequestStatus.FINISHED_IGNORED:
            # The ignored sequences are the sequences whose prompt lengths
            # are longer than the model's length cap. Therefore, the stop
            # reason should also be "length" as in OpenAI API.
            finish_reason = "length"
        else:
            finish_reason = None
        return finish_reason


@dataclass
class DecodingSchedulableRequest(SchedulableRequest):
    # inputs
    prompt: Optional[str] = None
    prompt_token_ids: List[int] = field(default_factory=list)
    sampling_params: Optional["SamplingParams"] = None
    eos_token_id: int = 0

    # outputs
    output_token_ids: List[int] = field(default_factory=list)
    output_text: str = ""

    num_computed_tokens: int = 0
    cached_all_token_ids: List[int] = field(default_factory=list)

    prompt_logprobs: Optional["PromptLogprobs"] = None
    output_logprobs: "SampleLogprobs" = field(default_factory=list)
    cumulative_logprob: float = 0.0

    # status
    status: RequestStatus = RequestStatus.WAITING
    stop_reason: Union[int, str, None] = None
    is_prefill: bool = True
    busy: bool = False
    vblock: Optional["BlockTable"] = None

    # intermediate variable
    # for model input
    input_tokens: List[int] = field(default_factory=list)
    input_positions: List[int] = field(default_factory=list)

    # for attention metadata
    token_chunk_size: int = 0
    seq_len: int = 0
    query_len: int = 0
    context_len: int = 0
    physical_block_ids: List[int] = field(default_factory=list)

    # for sampling metadata
    do_sample: bool = False
    generator: Optional[torch.Generator] = None
    prompt_logprobs_indices: Optional[List[int]] = None
    sample_logprobs_indices: Optional[List[int]] = None

    # for incremental detokenization
    prefix_offset: int = 0
    read_offset: int = 0
    tokens: Optional[List[str]] = None

    @property
    def finished(self):
        return RequestStatus.is_finished(self.status)

    def __post_init__(self):
        self.cached_all_token_ids = self.prompt_token_ids + self.output_token_ids

    def set_scheduled_ts(self, scheduled_ts):
        self.metrics.scheduled_ts = scheduled_ts
        self.metrics.waiting_time = self.metrics.scheduled_ts - self.metrics.arrival_ts
        if self.metrics.first_scheduled_ts is None:
            self.metrics.first_scheduled_ts = scheduled_ts

    @property
    def num_new_tokens(self):
        return self.token_chunk_size

    def get_output_text_to_return(self, buffer_length: int):
        truncate = buffer_length and not self.finished
        return self.output_text[:-buffer_length] if truncate else (
            self.output_text)

    def get_len(self):
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    def update_num_computed_tokens(self, num_new_computed_tokens: int):
        self.num_computed_tokens += num_new_computed_tokens
        assert self.num_computed_tokens <= self.get_len(), (
            self.num_computed_tokens, self.get_len())
        if self.get_num_uncomputed_tokens() == 0:
            self.is_prefill = False

    def get_num_uncomputed_tokens(self) -> int:
        return self.get_len() - self.get_num_computed_tokens()

    def get_num_computed_tokens(self) -> int:
        return self.num_computed_tokens

    def append_token_id(
        self,
        token_id: int,
        logprobs: Dict[int, "Logprob"],
    ) -> None:
        assert token_id in logprobs
        self.output_logprobs.append(logprobs)

        logprob = logprobs[token_id].logprob

        self.output_token_ids.append(token_id)
        self.cached_all_token_ids.append(token_id)
        self.cumulative_logprob += logprob

    def get_token_ids(self) -> List[int]:
        return self.cached_all_token_ids

    def get_output_len(self) -> int:
        return len(self.output_token_ids)

    def get_output_token_ids(self) -> List[int]:
        return self.output_token_ids

    def get_last_token_id(self) -> int:
        if not self.output_token_ids:
            return self.prompt_token_ids[-1]
        return self.output_token_ids[-1]

    def len_prompt_token_ids(self):
        return len(self.prompt_token_ids)

    def get_num_new_tokens(self):
        if self.is_prefill:
            return self.get_num_uncomputed_tokens()
        else:
            return 1

    def reset_state_for_recompute(self):
        self.status = RequestStatus.WAITING
        self.num_computed_tokens = 0
        self.vblock = None