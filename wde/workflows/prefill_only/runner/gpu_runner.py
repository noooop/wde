import torch

from wde.logger import init_logger
from wde.workflows.core.runner.gpu_runner import GPURunner
from wde.workflows.core.schema.execute_io import ExecuteOutput
from wde.workflows.prefill_only.schema.execute_io import ModelInputForGPU

logger = init_logger(__name__)


class PrefillOnlyGPURunner(GPURunner):

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: ModelInputForGPU,
    ) -> ExecuteOutput:
        return self.model(**model_input.to_dict())
