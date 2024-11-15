from typing import List

from wde.tasks.reranker.schema.engine_io import RerankerRequestOutput
from wde.tasks.reranker.schema.execute_io import RerankerExecuteOutput
from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.processor.output_processor import OutputProcessor
from wde.workflows.prefill_only.schema.engine_io import \
    PrefillOnlySchedulerOutput


class RerankerOutputProcessor(OutputProcessor):

    def __init__(self):
        pass

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls()

    def __call__(
            self, scheduler_output: PrefillOnlySchedulerOutput,
            execute_output: RerankerExecuteOutput
    ) -> List[RerankerRequestOutput]:
        scores = execute_output.scores.view(-1, ).numpy().tolist()
        request_outputs = []
        for i, request in enumerate(scheduler_output.scheduled_requests):
            prompt_token_ids = request.inputs.prompt_token_ids
            request_outputs.append(
                RerankerRequestOutput(request_id=request.request_id,
                                      metrics=request.metrics,
                                      prompt_token_ids=prompt_token_ids,
                                      finished=True,
                                      score=float(scores[i])))
        return request_outputs
