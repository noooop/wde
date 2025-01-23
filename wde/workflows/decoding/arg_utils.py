from dataclasses import dataclass
from typing import Optional

from wde.logger import init_logger
from wde.workflows.core.arg_utils import EngineArgs
from wde.workflows.decoding.config import (CacheConfig, DecodingEngineConfig,
                                           DecodingModelConfig,
                                           DecodingOffloadingSchedulerConfig,
                                           DecodingSchedulerConfig,
                                           EngineConfig)

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class DecodingEngineArgs(EngineArgs):
    # model_config
    kv_cache_dtype: str = 'auto'
    max_logprobs: int = 20  # Default value for OpenAI Chat Completions API

    # cache_config
    block_size: int = 16
    enable_prefix_caching: bool = False
    block_allocator: Optional[str] = None
    swap_space: int = 0  # GiB
    cpu_offload_gb: int = 0  # GiB
    gpu_memory_utilization: float = 0.90
    num_gpu_blocks_override: Optional[int] = None
    remote_kv_cache_server_name: Optional[str] = None
    watermark: float = 0.01

    # scheduler_config
    max_num_batched_tokens: Optional[int] = None
    max_num_requests: int = 256
    max_num_on_the_fly: Optional[int] = None
    scheduling: str = "async"
    preemption_mode: Optional[str] = None
    max_num_swap_in_blocks: Optional[int] = None
    max_num_swap_in_requests: Optional[int] = None

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    def create_engine_config(self, ) -> EngineConfig:
        engine_config = super().create_engine_config()

        assert self.cpu_offload_gb >= 0, (
            "CPU offload space must be non-negative"
            f", but got {self.cpu_offload_gb}")

        model_config = DecodingModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            max_logprobs=self.max_logprobs,
            disable_sliding_window=self.disable_sliding_window,
            served_model_name=self.served_model_name)

        cache_config = CacheConfig(
            block_size=self.block_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            swap_space=self.swap_space,
            cache_dtype=self.kv_cache_dtype,
            num_gpu_blocks_override=self.num_gpu_blocks_override,
            sliding_window=engine_config.model_config.get_sliding_window(),
            enable_prefix_caching=self.enable_prefix_caching,
            remote_kv_cache_server_name=self.remote_kv_cache_server_name,
            block_allocator=self.block_allocator,
            cpu_offload_gb=self.cpu_offload_gb,
            watermark=self.watermark)

        if self.swap_space > 0:
            scheduler_config = DecodingOffloadingSchedulerConfig(
                frieren_executor_max_workers=self.frieren_executor_max_workers,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_requests=self.max_num_requests,
                max_model_len=engine_config.model_config.max_model_len,
                preemption_mode=self.preemption_mode,
                max_num_on_the_fly=self.max_num_on_the_fly,
                scheduling=self.scheduling,
                block_size=self.block_size,
                max_num_swap_in_blocks=self.max_num_swap_in_blocks,
                max_num_swap_in_requests=self.max_num_swap_in_requests,
            )
        else:
            scheduler_config = DecodingSchedulerConfig(
                frieren_executor_max_workers=self.frieren_executor_max_workers,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_requests=self.max_num_requests,
                max_model_len=engine_config.model_config.max_model_len,
                preemption_mode=self.preemption_mode,
                max_num_on_the_fly=self.max_num_on_the_fly,
                scheduling=self.scheduling)

        return DecodingEngineConfig(model_config=model_config,
                                    cache_config=cache_config,
                                    scheduler_config=scheduler_config,
                                    device_config=engine_config.device_config,
                                    load_config=engine_config.load_config,
                                    sys_config=engine_config.sys_config,
                                    parallel_config=None)
