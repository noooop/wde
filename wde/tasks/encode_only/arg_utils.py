from dataclasses import dataclass
from typing import Optional

from wde.logger import init_logger
from wde.tasks.encode_only.config import (EncodeOnlyEngineConfig,
                                          PrefillOnlyParallelConfig,
                                          PrefillOnlySchedulerConfig)
from wde.workflows.core.arg_utils import EngineArgs
from wde.workflows.core.config import filter_unexpected_fields

logger = init_logger(__name__)


@filter_unexpected_fields
@dataclass
class EncodeOnlyEngineArgs(EngineArgs):
    # scheduler_config
    max_num_requests: int = 8
    max_num_batched_tokens: Optional[int] = None
    max_num_on_the_fly: Optional[int] = None
    scheduling: str = "async"
    waiting: Optional[float] = None

    # parallel_config
    data_parallel_size: int = 0

    def __post_init__(self):
        if self.tokenizer is None:
            self.tokenizer = self.model

    def create_engine_config(self) -> EncodeOnlyEngineConfig:
        engine_config = super().create_engine_config()

        scheduler_config = PrefillOnlySchedulerConfig(
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_requests=self.max_num_requests,
            max_model_len=engine_config.model_config.max_model_len,
            max_num_on_the_fly=self.max_num_on_the_fly,
            scheduling=self.scheduling,
            waiting=self.waiting)
        if self.data_parallel_size > 0:
            parallel_config = PrefillOnlyParallelConfig(
                data_parallel_size=self.data_parallel_size)
        else:
            parallel_config = None

        return EncodeOnlyEngineConfig(
            model_config=engine_config.model_config,
            scheduler_config=scheduler_config,
            device_config=engine_config.device_config,
            load_config=engine_config.load_config,
            sys_config=engine_config.sys_config,
            parallel_config=parallel_config)
