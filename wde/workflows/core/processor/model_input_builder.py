from abc import ABC, abstractmethod

from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.schema.engine_io import SchedulerOutput
from wde.workflows.core.schema.execute_io import ExecuteInput


class ModelInputBuilder(ABC):
    """
    scheduler_output = scheduler.schedule()
    SchedulerOutput  -> ModelInputBuilder -> ExecuteInput
    """

    @abstractmethod
    def __call__(self, scheduler_output: SchedulerOutput) -> ExecuteInput:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_engine(cls, engine: LLMEngine):
        raise NotImplementedError
