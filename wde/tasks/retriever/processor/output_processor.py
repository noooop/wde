from typing import List

from wde.tasks.encode_only.processor.output_processor import \
    EncodeOnlyOutputProcessor
from wde.tasks.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput
from wde.tasks.retriever.schema.engine_io import EmbeddingRequestOutput
from wde.tasks.retriever.schema.execute_io import RetrieverExecuteOutput


class RetrieverOutputProcessor(EncodeOnlyOutputProcessor):

    def __call__(
        self, scheduler_output: PrefillOnlySchedulerOutput,
        execute_output: RetrieverExecuteOutput
    ) -> List[EmbeddingRequestOutput]:
        request_outputs = []
        for i, request in enumerate(scheduler_output.scheduled_requests):
            prompt_token_ids = request.inputs.prompt_token_ids
            request_outputs.append(
                EmbeddingRequestOutput(request_id=request.request_id,
                                       metrics=request.metrics,
                                       prompt_token_ids=prompt_token_ids,
                                       finished=True,
                                       outputs=execute_output.embeddings[i]))
        return request_outputs
