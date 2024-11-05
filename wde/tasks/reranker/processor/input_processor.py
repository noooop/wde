import time
from typing import Optional, Sequence

from wde.tasks.core.llm_engine import LLMEngine
from wde.tasks.core.processor.input_processor import (InputProcessor,
                                                      RequestProcessor)
from wde.tasks.core.schema.engine_io import (Params, RequestMetrics,
                                             ValidationError)
from wde.tasks.prefill_only.processor.tokenizer import Tokenizer
from wde.tasks.prefill_only.schema.engine_io import (
    PrefillOnlyInput, PrefillOnlySchedulableRequest)
from wde.tasks.reranker.schema.engine_io import (Pairs, RerankerInputs,
                                                 RerankerRequest)


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
                                  arrival_time=arrival_time)
        return request


class RerankerRequestProcessor(RequestProcessor):

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine.tokenizer)

    def __call__(self,
                 request: RerankerRequest) -> PrefillOnlySchedulableRequest:
        text_pair = (request.inputs.query, request.inputs.passage)
        prompt_token_ids = self.tokenizer.encode(text_pair)
        schedulable_request = PrefillOnlySchedulableRequest(
            request_id=request.request_id,
            inputs=PrefillOnlyInput(prompt_token_ids=prompt_token_ids,
                                    prompt=None),
            metrics=RequestMetrics(arrival_ts=request.arrival_time))
        return schedulable_request
