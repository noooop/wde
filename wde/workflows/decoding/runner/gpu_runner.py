from typing import Optional

import torch

from wde.logger import init_logger
from wde.workflows.core.backends.attention import AttentionBackend
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.runner.gpu_runner import GPURunner
from wde.workflows.decoding.backends.sampling.logits_processor import \
    LogitsProcessor
from wde.workflows.decoding.backends.sampling.sampler import Sampler
from wde.workflows.decoding.backends.sampling.sampling_metadata import \
    SamplingMetadata
from wde.workflows.decoding.schema.execute_io import (DecodingModelInput,
                                                      SamplerOutput)

logger = init_logger(__name__)


class GPUDecodingRunner(GPURunner):

    def __init__(
        self,
        engine_config: EngineConfig,
        attn_backend: AttentionBackend,
    ):
        super().__init__(engine_config, attn_backend)

        self.logits_processor = LogitsProcessor(
            engine_config.model_config.hf_config.vocab_size)
        self.sampler = Sampler()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        logits = self.logits_processor(self.model.lm_head, hidden_states,
                                       sampling_metadata)
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    @torch.inference_mode()
    def execute_model(
        self,
        model_input: DecodingModelInput,
    ) -> SamplerOutput:

        hidden_states = self.model(input_ids=model_input.input_tokens,
                                   positions=model_input.input_positions,
                                   kv_caches=model_input.kv_caches,
                                   attn_metadata=model_input.attn_metadata)

        logits = self.compute_logits(hidden_states,
                                     model_input.sampling_metadata)

        output = self.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        return output