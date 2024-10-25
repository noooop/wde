import time
from typing import List

from wde.tasks.core.llm_engine import LLMEngine
from wde.tasks.core.processor.output_processor import (OutputProcessor,
                                                       RequestOutput)
from wde.tasks.encode_only.schema.execute_io import EncodeOnlyExecuteOutput
from wde.tasks.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput
from wde.tasks.retriever.schema.engine_io import EmbeddingRequestOutput


class EncodeOnlyOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(
            self, scheduler_output: PrefillOnlySchedulerOutput,
            execute_output: EncodeOnlyExecuteOutput) -> List[RequestOutput]:
        if execute_output.pooled_output is not None:
            request_outputs = []
            for i, request in enumerate(scheduler_output.scheduled_requests):
                prompt_token_ids = request.inputs.prompt_token_ids
                request.metrics.finish_ts = time.perf_counter()
                request.metrics.delay = request.metrics.finish_ts - request.metrics.first_scheduled_ts
                request_outputs.append(
                    EmbeddingRequestOutput(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        metrics=request.metrics,
                        prompt_token_ids=prompt_token_ids,
                        finished=True,
                        outputs=execute_output.pooled_output[i]))
            return request_outputs
        else:
            request_outputs = []
            for i, request in enumerate(scheduler_output.scheduled_requests):
                prompt_token_ids = request.inputs.prompt_token_ids
                request.metrics.finish_ts = time.perf_counter()
                request.metrics.delay = request.metrics.finish_ts - request.metrics.first_scheduled_ts
                request_outputs.append(
                    EmbeddingRequestOutput(
                        request_id=request.request_id,
                        arrival_time=request.arrival_time,
                        metrics=request.metrics,
                        prompt_token_ids=prompt_token_ids,
                        finished=True,
                        outputs=execute_output.last_hidden_states[i]))
            return request_outputs
