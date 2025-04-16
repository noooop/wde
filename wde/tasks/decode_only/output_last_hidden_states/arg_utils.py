from dataclasses import dataclass
from typing import Optional

from wde.logger import init_logger
from wde.tasks.decode_only.output_last_hidden_states.config import (
    DecodeOnlyEmbeddingSchedulerConfig, DecodeOnlyEngineConfig,
    DecodeOnlyModelConfig, DecodeOnlySchedulerConfig,
    PrefillOnlyParallelConfig)
from wde.workflows.core.arg_utils import EngineArgs
from wde.workflows.core.config import filter_unexpected_fields

logger = init_logger(__name__)


@filter_unexpected_fields
@dataclass
class DecodeOnlyOutputLastHiddenStatesEngineArgs(EngineArgs):
    # model_config
    output_last_hidden_states: bool = False
    enable_bidirectional: bool = False

    # scheduler_config
    max_num_requests: int = 8
    max_num_on_the_fly: Optional[int] = None
    scheduling: str = "async"
    waiting: Optional[float] = None
    max_num_batched_tokens: Optional[int] = None

    # parallel_config
    data_parallel_size: int = 0

    def create_engine_config(self) -> DecodeOnlyEngineConfig:
        engine_config = super().create_engine_config()

        model_config = DecodeOnlyModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            disable_sliding_window=self.disable_sliding_window,
            served_model_name=self.served_model_name,
            output_last_hidden_states=self.output_last_hidden_states,
            enable_bidirectional=self.enable_bidirectional)

        if model_config.output_last_hidden_states:
            scheduler_config = DecodeOnlyEmbeddingSchedulerConfig(
                frieren_executor_max_workers=self.frieren_executor_max_workers,
                max_num_batched_tokens=self.max_num_batched_tokens,
                max_num_requests=self.max_num_requests,
                max_model_len=model_config.max_model_len,
                max_num_on_the_fly=self.max_num_on_the_fly,
                scheduling=self.scheduling,
                waiting=self.waiting)
        else:
            scheduler_config = DecodeOnlySchedulerConfig()

        if (model_config.output_last_hidden_states
                and self.data_parallel_size > 0):
            parallel_config = PrefillOnlyParallelConfig(
                data_parallel_size=self.data_parallel_size)
        else:
            parallel_config = None

        return DecodeOnlyEngineConfig(
            model_config=model_config,
            scheduler_config=scheduler_config,
            device_config=engine_config.device_config,
            load_config=engine_config.load_config,
            sys_config=engine_config.sys_config,
            parallel_config=parallel_config)
