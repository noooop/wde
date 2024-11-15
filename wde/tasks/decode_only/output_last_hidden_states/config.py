from dataclasses import fields
from typing import Optional

from wde.logger import init_logger
from wde.workflows.core.config import (EngineConfig, ModelConfig,
                                       SchedulerConfig)
from wde.workflows.prefill_only.config import (PrefillOnlyParallelConfig,
                                               PrefillOnlySchedulerConfig)

logger = init_logger(__name__)

_GB = 1 << 30


class DecodeOnlyModelConfig(ModelConfig):

    def __init__(self,
                 output_last_hidden_states: bool = False,
                 enable_bidirectional: bool = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.output_last_hidden_states = output_last_hidden_states
        self.enable_bidirectional = enable_bidirectional

        self._verify_parameters()

    def _verify_parameters(self) -> None:
        if self.enable_bidirectional is None:
            if hasattr(self.hf_config, "enable_bidirectional"):
                self.enable_bidirectional = self.hf_config.enable_bidirectional
            else:
                self.enable_bidirectional = False

        if self.enable_bidirectional:
            self.output_last_hidden_states = True


class DecodeOnlySchedulerConfig(SchedulerConfig):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_last_hidden_states = False


class DecodeOnlyEmbeddingSchedulerConfig(DecodeOnlySchedulerConfig,
                                         PrefillOnlySchedulerConfig):

    def __init__(self, *args, **kwargs) -> None:
        PrefillOnlySchedulerConfig.__init__(self, *args, **kwargs)
        self.output_last_hidden_states = True


class DecodeOnlyEngineConfig(EngineConfig):
    model_config: DecodeOnlyModelConfig
    scheduler_config: DecodeOnlySchedulerConfig
    parallel_config: Optional[PrefillOnlyParallelConfig]

    def to_dict(self):
        """Return the configs as a dictionary, for use in **kwargs.
        """
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))

    def log_config(self):
        from wde.version import __version__ as VERSION
        if self.scheduler_config.output_last_hidden_states:
            logger.info(
                "Initializing an Encode Only engine (v%s) with config: "
                "model=%r, tokenizer=%r, "
                "tokenizer_mode=%s, "
                "trust_remote_code=%s, dtype=%s, max_seq_len=%d, "
                "download_dir=%r, load_format=%s, "
                "device_config=%s, served_model_name=%s, "
                "max_num_on_the_fly=%d, scheduling=%s)", VERSION,
                self.model_config.model, self.model_config.tokenizer,
                self.model_config.tokenizer_mode,
                self.model_config.trust_remote_code, self.model_config.dtype,
                self.model_config.max_model_len, self.load_config.download_dir,
                self.load_config.load_format, self.device_config.device,
                self.model_config.served_model_name,
                self.scheduler_config.max_num_on_the_fly,
                self.scheduler_config.scheduling)
            if self.parallel_config is not None:
                logger.info("Parallel config: data_parallel_size=%d",
                            self.parallel_config.data_parallel_size)
