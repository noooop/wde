from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Iterable, List, Optional, Set, Union

from wde.logger import init_logger
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.core.schema.engine_io import (Request, RequestOutput,
                                                 SchedulerOutput)

logger = init_logger(__name__)


class Scheduler(ABC):
    support_scheduling: List[str] = []

    def __init__(
        self,
        engine_config: Optional[EngineConfig],
        request_processor: Optional[RequestProcessor],
    ) -> None:
        self.engine_config = engine_config
        self.scheduler_config = getattr(engine_config, "scheduler_config",
                                        None)
        self.request_processor = request_processor

        self.waiting: Deque[Request] = deque()

        self.requests: Set[str] = set()
        self.aborted_requests: Set[str] = set()

    @classmethod
    def from_engine(cls, engine) -> "Scheduler":
        raise NotImplementedError

    def add_request(self, request: Request) -> None:
        if (request.request_id in self.requests
                or request.request_id in self.aborted_requests):
            logger.warning("[%s] request_id conflict", request.request_id)
            return

        self.waiting.append(request)
        self.requests.add(request.request_id)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        if isinstance(request_id, str):
            request_id = (request_id, )
        request_ids = set(request_id)

        self.requests -= request_ids
        self.aborted_requests |= request_ids

    def actual_abort_request(self, request_id: str):
        self.aborted_requests.remove(request_id)

    def remove_abort_request(
            self, request_outputs: List[RequestOutput]) -> List[RequestOutput]:
        if len(self.aborted_requests) == 0:
            return request_outputs

        current_ids = set(request.request_id for request in request_outputs)
        need_abort = self.aborted_requests & current_ids

        if len(need_abort) == 0:
            return request_outputs

        request_outputs = [
            request for request in request_outputs
            if request.request_id not in need_abort
        ]
        for request_id in need_abort:
            self.actual_abort_request(request_id)

        return request_outputs

    def has_unfinished_requests(self) -> bool:
        return len(self.requests) != 0

    def get_num_unfinished_requests(self) -> int:
        return len(self.requests)

    @abstractmethod
    def schedule(self) -> SchedulerOutput:
        raise NotImplementedError

    def free_finished_request(self, request_outputs: List[RequestOutput]):
        finished_request_ids = set(request.request_id
                                   for request in request_outputs
                                   if request.finished)
        self.requests -= finished_request_ids

    def clear(self):
        self.waiting.clear()
        self.requests.clear()
        self.aborted_requests.clear()

    def join(self):
        pass