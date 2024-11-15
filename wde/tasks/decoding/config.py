from dataclasses import dataclass, fields
from typing import Optional

from wde.logger import init_logger
from wde.tasks.core.config import CacheConfig, EngineConfig, ModelConfig

logger = init_logger(__name__)

_GB = 1 << 30


class DecodingModelConfig(ModelConfig):

    def __init__(self, max_logprobs: int = 20, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_logprobs = max_logprobs


class DecodingSchedulerConfig:
    supported_scheduling = ["sync", "simple_async", "async", "double_buffer"]

    def __init__(self,
                 max_num_batched_tokens: Optional[int],
                 max_num_seqs: int,
                 max_model_len: int,
                 preemption_mode: Optional[str] = None,
                 max_num_on_the_fly: Optional[int] = None,
                 scheduling: str = "async") -> None:
        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            self.max_num_batched_tokens = 512

        logger.info(
            "Chunked prefill is enabled with max_num_batched_tokens=%d.",
            self.max_num_batched_tokens)

        if max_num_on_the_fly is None:
            if scheduling == "double_buffer":
                self.max_num_on_the_fly = 3
            else:
                self.max_num_on_the_fly = 2
        else:
            self.max_num_on_the_fly = max_num_on_the_fly

        self.max_num_seqs = max_num_seqs
        self.max_model_len = max_model_len
        self.preemption_mode = preemption_mode
        self.scheduling = scheduling
        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_num_seqs:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_num_seqs "
                f"({self.max_num_seqs}).")

        if self.max_num_on_the_fly < 2:
            raise ValueError(
                f"max_num_on_the_fly {self.max_num_on_the_fly} must "
                "be greater than 1")


@dataclass
class DecodingEngineConfig(EngineConfig):

    model_config: DecodingModelConfig
    cache_config: Optional[CacheConfig] = None
    scheduler_config: DecodingSchedulerConfig

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))

    def log_config(self):
        from wde.version import __version__ as VLLM_VERSION
        logger.info(
            "Initializing an LLM engine (v%s) with config: "
            "model=%r, tokenizer=%r, "
            "skip_tokenizer_init=%s, tokenizer_mode=%s, revision=%s, "
            "rope_scaling=%r, rope_theta=%r, tokenizer_revision=%s, "
            "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
            "download_dir=%r, load_format=%s, "
            "quantization=%s, kv_cache_dtype=%s, "
            "quantization_param_path=%s, device_config=%s, "
            "seed=%d, served_model_name=%s, "
            "enable_prefix_caching=%s, scheduling=%s)",
            VLLM_VERSION,
            self.model_config.model,
            self.model_config.tokenizer,
            self.model_config.skip_tokenizer_init,
            self.model_config.tokenizer_mode,
            self.model_config.revision,
            self.model_config.rope_scaling,
            self.model_config.rope_theta,
            self.model_config.tokenizer_revision,
            self.model_config.trust_remote_code,
            self.model_config.dtype,
            self.model_config.max_model_len,
            self.load_config.download_dir,
            self.load_config.load_format,
            self.model_config.quantization,
            self.cache_config.cache_dtype,
            self.model_config.quantization_param_path,
            self.device_config.device,
            self.model_config.seed,
            self.model_config.served_model_name,
            self.cache_config.enable_prefix_caching,
            self.scheduler_config.scheduling,
        )
