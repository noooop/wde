import time
from typing import Optional, Sequence

from wde.tasks.reranker.schema.engine_io import (Pairs, RerankerInputs,
                                                 RerankerRequest)
from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.processor.input_processor import (InputProcessor,
                                                          RequestProcessor)
from wde.workflows.core.schema.engine_io import (Params, RequestMetrics,
                                                 TextOnlyInputs,
                                                 TextSchedulableRequest,
                                                 ValidationError)
from wde.workflows.prefill_only.processor.tokenizer import Tokenizer


class RerankerInputProcessor(InputProcessor):

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self,
                 request_id: str,
                 inputs: Optional[RerankerInputs] = None,
                 params: Optional[Params] = None,
                 arrival_time: Optional[float] = None) -> RerankerRequest:
        if not arrival_time:
            arrival_time = time.perf_counter()

        if isinstance(inputs, Sequence):
            if len(inputs) != 2:
                raise ValidationError("Reranker model input must be pairs.")
            inputs = Pairs(query=inputs[0], passage=inputs[1])

        request = RerankerRequest(request_id=str(request_id),
                                  inputs=inputs,
                                  params=params,
                                  arrival_time=arrival_time)
        return request


class RerankerRequestProcessor(RequestProcessor):

    def __init__(self, tokenizer: Tokenizer, record_metrics=False):
        self.tokenizer = tokenizer
        self.record_metrics = record_metrics

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.tokenizer,
                   engine.engine_config.sys_config.record_metrics)

    def __call__(self, request: RerankerRequest) -> TextSchedulableRequest:
        text_pair = (request.inputs.query, request.inputs.passage)
        prompt_token_ids = self.tokenizer.encode(text_pair)

        if self.record_metrics:
            metrics = RequestMetrics(arrival_ts=request.arrival_time)
        else:
            metrics = None

        schedulable_request = TextSchedulableRequest(
            request_id=request.request_id,
            inputs=TextOnlyInputs(prompt_token_ids=prompt_token_ids,
                                  prompt=None),
            params=request.params,
            arrival_time=request.arrival_time,
            metrics=metrics)
        return schedulable_request
