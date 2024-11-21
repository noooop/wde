import warnings
from typing import List, Optional

import torch
import torch.distributed
import torch.nn as nn
from vllm.platforms import current_platform
from vllm.utils import DeviceMemoryProfiler, is_pin_memory_available

from wde.backends.models.utils import set_cpu_offload_max_bytes
from wde.logger import init_logger
from wde.workflows.core.config import DeviceConfig, LoadConfig
from wde.workflows.decoding.backends.attention import \
    DecodeOnlyAttentionBackend
from wde.workflows.decoding.backends.logits_processor import LogitsProcessor
from wde.workflows.decoding.backends.sampler import Sampler
from wde.workflows.decoding.backends.sampling_metadata import SamplingMetadata
from wde.workflows.decoding.backends.sampling_params import SamplingParams
from wde.workflows.decoding.backends.sequence import (SequenceData,
                                                      SequenceGroupMetadata)
from wde.workflows.decoding.config import (CacheConfig, DecodingModelConfig,
                                           DecodingSchedulerConfig)
from wde.workflows.decoding.schema.execute_io import (
    DecodingModelInputForGPUWithSamplingMetadata, SamplerOutput)

logger = init_logger(__name__)


class GPUModelRunner:

    def __init__(self,
                 model_config: DecodingModelConfig,
                 scheduler_config: DecodingSchedulerConfig,
                 device_config: DeviceConfig,
                 cache_config: CacheConfig,
                 load_config: LoadConfig,
                 attn_backend: DecodeOnlyAttentionBackend,
                 kv_cache_dtype: Optional[str] = "auto"):
        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.device_config = device_config
        self.cache_config = cache_config
        self.load_config = load_config
        self.attn_backend = attn_backend

        self.device = self.device_config.device
        self.pin_memory = is_pin_memory_available()

        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size

        # Lazy initialization
        self.model: nn.Module  # Set after load_model

        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))

        self.logits_processor = LogitsProcessor(
            self.model_config.hf_config.vocab_size)
        self.sampler = Sampler()

    def load_model(self) -> None:
        from wde.backends.loader.loader import (get_model_loader,
                                                initialize_model)

        logger.info("Starting to load model %s...", self.model_config.model)
        with DeviceMemoryProfiler() as m:
            loader = get_model_loader(self.load_config)
            self.model = initialize_model(model_config=self.model_config,
                                          load_config=self.load_config,
                                          device_config=self.device_config,
                                          cache_config=self.cache_config,
                                          attn_backend=self.attn_backend)

            loader.load_model(self.model,
                              model_config=self.model_config,
                              device_config=self.device_config)

        if self.kv_cache_dtype == "fp8" and current_platform.is_hpu():
            # Currently only ROCm accepts kv-cache scaling factors
            # via quantization_param_path and this will be deprecated
            # in the future.
            if self.model_config.quantization_param_path is not None:
                if callable(getattr(self.model, "load_kv_cache_scales", None)):
                    warnings.warn(
                        "Loading kv cache scaling factor from JSON is "
                        "deprecated and will be removed. Please include "
                        "kv cache scaling factors in the model checkpoint.",
                        FutureWarning,
                        stacklevel=2)
                    self.model.load_kv_cache_scales(
                        self.model_config.quantization_param_path)
                    logger.info("Loaded KV cache scaling factors from %s",
                                self.model_config.quantization_param_path)
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        "model %s does not support loading scaling factors.",
                        self.model.__class__)
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors "
                    "provided. Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!")

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_requests = self.scheduler_config.max_num_requests

        seqs: List[SequenceGroupMetadata] = []
        batch_size = 0
        for group_id in range(max_num_requests):
            seq_len = (max_num_batched_tokens // max_num_requests +
                       (group_id < max_num_batched_tokens % max_num_requests))
            batch_size += seq_len

            seq_data = SequenceData.from_token_counts((0, seq_len))

            # Having more tokens is over-conservative but otherwise fine
            assert len(seq_data.prompt_token_ids) >= seq_len, (
                f"Expected at least {seq_len} dummy tokens for profiling, "
                f"but got: {len(seq_data.prompt_token_ids)}")

            seq = SequenceGroupMetadata(request_id=str(group_id),
                                        is_prompt=True,
                                        seq_data={group_id: seq_data},
                                        sampling_params=sampling_params,
                                        block_tables=None)
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers()
        kv_caches = [None] * num_layers
        model_input = self.prepare_model_input(seqs)
        model_input.to("cuda")
        self.execute_model(model_input, kv_caches)
        torch.cuda.synchronize()
        return

    @property
    def vocab_size(self) -> int:
        return self.model_config.get_vocab_size()

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
        model_input: DecodingModelInputForGPUWithSamplingMetadata,
        kv_caches: List[torch.Tensor],
    ) -> SamplerOutput:
        hidden_states = self.model(input_ids=model_input.input_tokens,
                                   positions=model_input.input_positions,
                                   kv_caches=kv_caches,
                                   attn_metadata=model_input.attn_metadata)

        logits = self.compute_logits(hidden_states,
                                     model_input.sampling_metadata)

        output = self.sample(
            logits=logits,
            sampling_metadata=model_input.sampling_metadata,
        )

        return output
