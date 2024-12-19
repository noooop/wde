import traceback
from abc import ABC, abstractmethod

import torch
from vllm.platforms import current_platform

from wde.logger import init_logger
from wde.utils import lazy_import
from wde.workflows.core.backends.attention import AttentionBackend
from wde.workflows.core.backends.utils import set_random_seed
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.schema.execute_io import ExecuteInput, ExecuteOutput
from wde.workflows.core.workflow import Workflow

logger = init_logger(__name__)


class WorkerBase(ABC):

    @abstractmethod
    def init_device(self):
        raise NotImplementedError

    @abstractmethod
    def load_model(self):
        raise NotImplementedError

    @abstractmethod
    def __call__(self, execute_input: ExecuteInput):
        raise NotImplementedError

    @abstractmethod
    def non_blocking_h2d(self, execute_input: ExecuteInput):
        raise NotImplementedError

    @abstractmethod
    def non_blocking_d2h(self, execute_output: ExecuteOutput):
        raise NotImplementedError

    @abstractmethod
    def deferred_h2d(self, execute_input: ExecuteInput):
        raise NotImplementedError

    @abstractmethod
    def deferred_d2h(self, execute_output: ExecuteOutput):
        raise NotImplementedError


class GPUWorker(WorkerBase):

    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: AttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.attn_backend = attn_backend

        if engine_config.model_config.trust_remote_code:
            from vllm.utils import init_cached_hf_modules
            init_cached_hf_modules()

        self.runner = lazy_import(workflow.Runer)(engine_config=engine_config,
                                                  attn_backend=attn_backend)

    def init_device(self) -> None:
        device_config = self.engine_config.device_config
        if device_config.device.type == "cuda":
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)

            _check_if_gpu_supports_dtype(self.engine_config.model_config.dtype)
            torch.cuda.empty_cache()
            self.init_gpu_memory = torch.cuda.mem_get_info()[0]
        else:
            raise RuntimeError(
                f"Not support device type: {device_config.device}")
        set_random_seed(self.engine_config.model_config.seed)

    def load_model(self):
        self.runner.load_model()

    @torch.inference_mode
    def __call__(self, execute_input: ExecuteInput) -> ExecuteOutput:
        try:
            output = self.runner.execute_model(execute_input)
        except Exception:
            traceback.print_exc()
        return output

    def non_blocking_h2d(self, execute_input: ExecuteInput):
        model_input = execute_input.model_input
        return model_input.to("cuda", non_blocking=True)

    def non_blocking_d2h(self, execute_output: ExecuteOutput):
        return execute_output.to("cpu", non_blocking=True)

    def deferred_h2d(self, execute_input: ExecuteInput):
        model_input = execute_input.model_input
        return model_input.deferred_to("cuda", non_blocking=True)

    def deferred_d2h(self, execute_output: ExecuteOutput):
        return execute_output.deferred_to("cpu", non_blocking=True)


def _check_if_gpu_supports_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        compute_capability = current_platform.get_device_capability()
        if compute_capability[0] < 8:
            gpu_name = torch.cuda.get_device_name()
            raise ValueError(
                "Bfloat16 is only supported on GPUs with compute capability "
                f"of at least 8.0. Your {gpu_name} GPU has compute capability "
                f"{compute_capability[0]}.{compute_capability[1]}. "
                "You can use float16 instead by explicitly setting the"
                "`dtype` flag in CLI, for example: --dtype=half.")
