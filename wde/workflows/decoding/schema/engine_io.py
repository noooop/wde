from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from wde.workflows.core.schema.engine_io import (RequestMetrics, RequestOutput,
                                                 SchedulableRequest,
                                                 SchedulerOutput)
from wde.workflows.decoding.backends.sequence import (PromptLogprobs,
                                                      SampleLogprobs,
                                                      SequenceGroup,
                                                      SequenceGroupMetadata,
                                                      SequenceStatus)


@dataclass
class DecodingSchedulableRequest(SchedulableRequest):
    seq_group: Optional[SequenceGroup] = None
    token_chunk_size: int = 0
    busy = False

    def set_scheduled_ts(self, scheduled_ts):
        self.metrics.scheduled_ts = scheduled_ts
        self.metrics.waiting_time = self.metrics.scheduled_ts - self.metrics.arrival_ts
        if self.metrics.first_scheduled_ts is None:
            self.metrics.first_scheduled_ts = scheduled_ts

    @property
    def num_new_tokens(self):
        return self.token_chunk_size


@dataclass
class DecodingSchedulerOutput(SchedulerOutput):
    scheduled_requests: List[DecodingSchedulableRequest]

    num_prefill_groups: int
    num_batched_tokens: int

    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]

    ignored_requests: List[DecodingSchedulableRequest]

    running_queue_size: int
    preempted: int

    seq_group_metadata_list: Optional[List[SequenceGroupMetadata]] = None

    def is_empty(self) -> bool:
        # NOTE: We do not consider the ignored sequence groups.
        return (not self.scheduled_requests and not self.blocks_to_swap_in
                and not self.blocks_to_swap_out and not self.blocks_to_copy)


@dataclass
class CompletionOutput:
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    index: int
    text: str
    token_ids: Tuple[int, ...]
    cumulative_logprob: Optional[float]
    logprobs: Optional[SampleLogprobs]
    finish_reason: Optional[str] = None
    stop_reason: Union[int, str, None] = None

    def finished(self) -> bool:
        return self.finish_reason is not None

    def __repr__(self) -> str:
        return (f"CompletionOutput(index={self.index}, "
                f"text={self.text!r}, "
                f"token_ids={self.token_ids}, "
                f"cumulative_logprob={self.cumulative_logprob}, "
                f"logprobs={self.logprobs}, "
                f"finish_reason={self.finish_reason}, "
                f"stop_reason={self.stop_reason})")


class DecodingRequestOutput(RequestOutput):

    def __init__(
        self,
        request_id: str,
        prompt: Optional[str],
        prompt_token_ids: List[int],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics

    @classmethod
    def from_seq_group(
            cls,
            request: DecodingSchedulableRequest) -> "DecodingRequestOutput":
        seq_group = request.seq_group

        if seq_group.sampling_params is None:
            raise ValueError(
                "Sampling parameters are missing for a CompletionRequest.")
        seqs = seq_group.get_seqs()
        if len(seqs) == 1:
            top_n_seqs = seqs
        else:
            n = seq_group.sampling_params.n
            sorting_key = lambda seq: seq.get_cumulative_logprob()
            sorted_seqs = sorted(seqs, key=sorting_key, reverse=True)
            top_n_seqs = sorted_seqs[:n]

        include_logprobs = seq_group.sampling_params.logprobs is not None
        text_buffer_length = seq_group.sampling_params.output_text_buffer_length
        outputs = [
            CompletionOutput(
                seqs.index(seq),
                seq.get_output_text_to_return(text_buffer_length),
                seq.get_output_token_ids(),
                seq.get_cumulative_logprob() if include_logprobs else None,
                seq.output_logprobs if include_logprobs else None,
                SequenceStatus.get_finished_reason(seq.status),
                seq.stop_reason) for seq in top_n_seqs
        ]

        prompt = seq_group.prompt
        prompt_token_ids = seq_group.prompt_token_ids
        prompt_logprobs = seq_group.prompt_logprobs
        finished = seq_group.is_finished()
        return cls(request.request_id, prompt, prompt_token_ids,
                   prompt_logprobs, outputs, finished, request.metrics)

    def __repr__(self) -> str:
        return (f"RequestOutput(request_id={self.request_id}, "
                f"prompt={self.prompt!r}, "
                f"prompt_token_ids={self.prompt_token_ids}, "
                f"prompt_logprobs={self.prompt_logprobs}, "
                f"outputs={self.outputs}, "
                f"finished={self.finished}, "
                f"metrics={self.metrics}, ")
