from typing import cast

import torch

from wde.logger import init_logger
from wde.workflows.core.runner.gpu_runner import GPURunner
from wde.workflows.core.schema.execute_io import ExecuteInput, ExecuteOutput
from wde.workflows.prefill_only.schema.execute_io import ModelInputForGPU

logger = init_logger(__name__)


class PrefillOnlyGPURunner(GPURunner):

    @torch.inference_mode()
    def execute_model(
        self,
        execute_input: ExecuteInput,
    ) -> ExecuteOutput:

        model_input = cast(ModelInputForGPU, execute_input.model_input)
        main_stream = execute_input.main_stream

        with torch.cuda.stream(main_stream):
            output = self.model(**model_input.to_dict())

        return output
