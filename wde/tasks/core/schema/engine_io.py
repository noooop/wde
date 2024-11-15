from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union


class Params:
    pass


class Inputs:
    pass


@dataclass
class TextPrompt(Inputs):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


@dataclass
class TokensPrompt(Inputs):
    """Schema for a tokenized prompt."""

    prompt_token_ids: List[int]
    """A list of token IDs to pass to the model."""


@dataclass
class TextOnlyInputs(Inputs):
    prompt_token_ids: List[int]
    """The token IDs of the prompt."""

    prompt: Optional[str] = None
    """
    The original prompt text corresponding to the token IDs, if available.
    """


PromptInputs = Union[str, Dict, TextPrompt, TokensPrompt, TextOnlyInputs]


@dataclass
class Request:
    request_id: str
    arrival_time: float


@dataclass
class TextRequest(Request):
    inputs: Dict
    params: Optional[Params]


class ValidationError(ValueError):
    pass


@dataclass
class RequestMetrics:
    arrival_ts: Optional[float] = None

    first_scheduled_ts: Optional[float] = None
    latency_so_far: Optional[float] = None

    scheduled_ts: Optional[float] = None
    inference_begin_ts: Optional[float] = None
    inference_end_ts: Optional[float] = None
    finish_ts: Optional[float] = None

    scheduling_time: Optional[float] = None
    num_requests: Optional[int] = None
    num_batched_tokens: Optional[int] = None

    waiting_time: Optional[float] = None
    scheduling2inference: Optional[float] = None
    inference_time: Optional[float] = None
    latency: Optional[float] = None


@dataclass
class SchedulableRequest:
    request_id: str
    metrics: RequestMetrics = field(default_factory=RequestMetrics)

    @property
    def num_new_tokens(self):
        raise NotImplementedError

    def set_scheduled_ts(self, scheduled_ts):
        self.metrics.scheduled_ts = scheduled_ts
        self.metrics.waiting_time = self.metrics.scheduled_ts - self.metrics.arrival_ts
        if self.metrics.first_scheduled_ts is None:
            self.metrics.first_scheduled_ts = scheduled_ts


@dataclass
class TextSchedulableRequest(SchedulableRequest):
    inputs: Optional[TextOnlyInputs] = None
    params: Params = field(default_factory=Params)

    @property
    def num_new_tokens(self):
        return len(self.inputs.prompt_token_ids)


@dataclass
class SchedulerOutput:
    pass


@dataclass
class RequestOutput:
    request_id: str
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
    finished: bool = True
