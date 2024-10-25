import time
from typing import List

from wde.tasks.core.llm_engine import LLMEngine
from wde.tasks.core.processor.output_processor import OutputProcessor
from wde.tasks.prefill_only.schema.engine_io import PrefillOnlySchedulerOutput
from wde.tasks.reranker.schema.engine_io import RerankerRequestOutput
from wde.tasks.reranker.schema.execute_io import RerankerExecuteOutput


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
            request.metrics.finish_ts = time.perf_counter()
            request.metrics.delay = request.metrics.finish_ts - request.metrics.first_scheduled_ts
            request_outputs.append(
                RerankerRequestOutput(request_id=request.request_id,
                                      arrival_time=request.arrival_time,
                                      metrics=request.metrics,
                                      prompt_token_ids=prompt_token_ids,
                                      finished=True,
                                      score=float(scores[i])))
        return request_outputs
