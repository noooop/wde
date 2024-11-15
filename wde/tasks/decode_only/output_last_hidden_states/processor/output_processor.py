from typing import List

import torch

from wde.tasks.retriever.schema.engine_io import EmbeddingRequestOutput
from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.processor.output_processor import OutputProcessor
from wde.workflows.prefill_only.schema.engine_io import \
    PrefillOnlySchedulerOutput


class DecodeOnlyHiddenStatesOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(self, scheduler_output: PrefillOnlySchedulerOutput,
                 execute_output: torch.Tensor) -> List[EmbeddingRequestOutput]:

        request_outputs = []
        offset = 0
        for request in scheduler_output.scheduled_requests:
            prompt_token_ids = request.inputs.prompt_token_ids
            n_tokens = len(prompt_token_ids)
            request_outputs.append(
                EmbeddingRequestOutput(request_id=request.request_id,
                                       metrics=request.metrics,
                                       prompt_token_ids=prompt_token_ids,
                                       finished=True,
                                       outputs=execute_output[offset +
                                                              n_tokens - 1]))
            offset += n_tokens
        return request_outputs
