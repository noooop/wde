from typing import List

from wde.tasks.encode_only.schema.execute_io import EncodeOnlyExecuteOutput
from wde.tasks.retriever.schema.engine_io import EmbeddingRequestOutput
from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.processor.output_processor import (OutputProcessor,
                                                           RequestOutput)
from wde.workflows.prefill_only.schema.engine_io import \
    PrefillOnlySchedulerOutput


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
                request_outputs.append(
                    EmbeddingRequestOutput(
                        request_id=request.request_id,
                        metrics=request.metrics,
                        prompt_token_ids=prompt_token_ids,
                        finished=True,
                        outputs=execute_output.pooled_output[i]))
            return request_outputs
        else:
            request_outputs = []
            for i, request in enumerate(scheduler_output.scheduled_requests):
                prompt_token_ids = request.inputs.prompt_token_ids
                request_outputs.append(
                    EmbeddingRequestOutput(
                        request_id=request.request_id,
                        metrics=request.metrics,
                        prompt_token_ids=prompt_token_ids,
                        finished=True,
                        outputs=execute_output.last_hidden_states[i]))
            return request_outputs
