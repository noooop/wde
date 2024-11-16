from typing import Optional

from wde.logger import init_logger
from wde.workflows.core.config import ParallelConfig, SchedulerConfig

logger = init_logger(__name__)

_GB = 1 << 30


class PrefillOnlySchedulerConfig(SchedulerConfig):
    supported_scheduling = ["sync", "simple_async", "async", "double_buffer"]

    def __init__(self,
                 max_model_len: int,
                 max_num_batched_tokens: Optional[int] = None,
                 max_num_requests: Optional[int] = None,
                 max_num_on_the_fly: Optional[int] = None,
                 scheduling: str = "async",
                 waiting: Optional[float] = None,
                 *args,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.max_model_len = max_model_len
        self.max_num_requests: int = 0
        self.max_num_batched_tokens: int = 0
        self.scheduling = scheduling
        self.waiting = waiting

        if max_num_on_the_fly is None:
            if scheduling == "double_buffer":
                self.max_num_on_the_fly = 3
            else:
                self.max_num_on_the_fly = 2
        else:
            self.max_num_on_the_fly = max_num_on_the_fly

        self.set_args(max_num_batched_tokens, max_num_requests)

    def set_args(self,
                 max_num_batched_tokens: Optional[int] = None,
                 max_num_requests: Optional[int] = None):
        self.max_num_requests = max_num_requests

        if max_num_batched_tokens is not None:
            self.max_num_batched_tokens = max_num_batched_tokens
        else:
            self.max_num_batched_tokens = (self.max_model_len *
                                           self.max_num_requests)

        self._verify_args()

    def _verify_args(self) -> None:
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                "be greater than or equal to max_model_len "
                f"({self.max_model_len}).")

        if self.max_num_on_the_fly < 2:
            raise ValueError(
                f"max_num_on_the_fly {self.max_num_on_the_fly} must "
                "be greater than 1")

        if self.scheduling not in self.supported_scheduling:
            raise ValueError(f"scheduling {self.scheduling} must "
                             f"in {self.supported_scheduling}")

        if self.waiting is not None:
            if self.waiting < 0.:
                raise ValueError(
                    f"waiting {self.scheduling} must be positive.")


class PrefillOnlyParallelConfig(ParallelConfig):

    def __init__(
        self,
        data_parallel_size: int,
    ):
        self.data_parallel_size = data_parallel_size
