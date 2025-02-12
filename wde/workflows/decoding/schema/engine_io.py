from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from wde.workflows.core.schema.engine_io import (RequestMetrics, RequestOutput,
                                                 SchedulerOutput)

from .execute_io import PromptLogprobs, SampleLogprobs
from .request import DecodingSchedulableRequest, RequestStatus


@dataclass
class DecodingSchedulerOutput(SchedulerOutput):
    scheduled_requests: List[DecodingSchedulableRequest]
    ignored_requests: List[DecodingSchedulableRequest]

    num_batched_tokens: int
    num_requests: int

    def is_empty(self) -> bool:
        return not self.scheduled_requests

    @classmethod
    def create_empty(cls) -> "DecodingSchedulerOutput":
        return cls(scheduled_requests=[],
                   ignored_requests=[],
                   num_batched_tokens=0,
                   num_requests=0)


@dataclass
class CompletionOutput:
    index: int
    text: str
    token_ids: Sequence[int]
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
        prompt_token_ids: Optional[List[int]],
        prompt_logprobs: Optional[PromptLogprobs],
        outputs: List[CompletionOutput],
        finished: bool,
        metrics: Optional[RequestMetrics] = None,
        num_cached_tokens: Optional[int] = None,
        num_preempted: Optional[int] = None,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.num_cached_tokens = num_cached_tokens
        self.num_preempted = num_preempted

    @classmethod
    def from_request(
            cls,
            request: DecodingSchedulableRequest) -> "DecodingRequestOutput":

        sampling_params = request.sampling_params

        include_logprobs = sampling_params.logprobs is not None

        text_buffer_length = sampling_params.output_text_buffer_length

        output = CompletionOutput(
            index=0,
            text=request.get_output_text_to_return(text_buffer_length),
            token_ids=request.get_output_token_ids(),
            cumulative_logprob=request.cumulative_logprob
            if include_logprobs else None,
            logprobs=request.output_logprobs if include_logprobs else None,
            finish_reason=RequestStatus.get_finished_reason(request.status),
            stop_reason=request.stop_reason)

        outputs = [output]

        prompt = request.prompt
        prompt_token_ids = request.prompt_token_ids
        prompt_logprobs = request.prompt_logprobs
        finished = request.finished

        # The last output token is definitely not computed
        num_cached_tokens = max(
            request.num_computed_tokens - request.num_actual_computed_tokens,
            0)
        num_preempted = request.num_preempted

        return cls(request.request_id, prompt, prompt_token_ids,
                   prompt_logprobs, outputs, finished, request.metrics,
                   num_cached_tokens, num_preempted)
