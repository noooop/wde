import torch
import torch.nn as nn
from vllm.utils import DeviceMemoryProfiler

from wde.logger import init_logger
from wde.workflows.core.backends.attention import AttentionBackend
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.schema.execute_io import ExecuteInput, ExecuteOutput

logger = init_logger(__name__)


class GPURunner:

    def __init__(
        self,
        engine_config: EngineConfig,
        attn_backend: AttentionBackend,
    ):
        self.engine_config = engine_config
        self.attn_backend = attn_backend
        self.device = self.engine_config.device_config.device
        self.model: nn.Module

    def load_model(self) -> None:
        torch.set_default_dtype(self.engine_config.model_config.dtype)

        from wde.workflows.core.backends.loader.loader import (
            get_model_loader, initialize_model)

        logger.info("Starting to load model %s...",
                    self.engine_config.model_config.model)
        with DeviceMemoryProfiler() as m:
            loader = get_model_loader(self.engine_config.load_config)
            self.model = initialize_model(engine_config=self.engine_config,
                                          attn_backend=self.attn_backend)

            loader.load_model(self.model,
                              model_config=self.engine_config.model_config,
                              device_config=self.engine_config.device_config)

        self.model_memory_usage = m.consumed_memory
        logger.info("Loading model weights took %.4f GB",
                    self.model_memory_usage / float(2**30))

    @torch.inference_mode()
    def execute_model(
        self,
        execute_input: ExecuteInput,
    ) -> ExecuteOutput:
        raise NotImplementedError
