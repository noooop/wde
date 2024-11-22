import time
from dataclasses import dataclass, field
from typing import Optional, Set, cast

from wde.logger import init_logger
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.core.scheduler import Scheduler
from wde.workflows.prefill_only.config import PrefillOnlySchedulerConfig
from wde.workflows.prefill_only.schema.engine_io import (
    PrefillOnlySchedulerOutput, SchedulableRequest)

logger = init_logger(__name__)


@dataclass
class PrefillOnlySchedulingBudget:
    token_budget: int
    max_num_requests: int
    _curr_requests: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_request: int = 1):
        assert num_new_tokens != 0
        assert num_new_request != 0
        a = self.num_batched_tokens + num_new_tokens <= self.token_budget
        b = self.num_curr_request + num_new_request <= self.max_num_requests
        return a and b

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._curr_requests:
            return

        self._curr_requests.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_request(self):
        return len(self._curr_requests)


class PrefillOnlyScheduler(Scheduler):
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(
        self,
        engine_config: EngineConfig,
        request_processor: RequestProcessor,
    ) -> None:
        super().__init__(engine_config, request_processor)
        self.record_metrics = engine_config.sys_config.record_metrics

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config, engine.request_processor)

    def schedule(self) -> Optional[PrefillOnlySchedulerOutput]:
        if not self.waiting:
            return None

        scheduler_config = cast(PrefillOnlySchedulerConfig,
                                self.scheduler_config)

        if self.record_metrics:
            scheduling_begin_ts = time.perf_counter()

        budget = PrefillOnlySchedulingBudget(
            token_budget=scheduler_config.max_num_batched_tokens,
            max_num_requests=scheduler_config.max_num_requests,
        )

        waiting = scheduler_config.waiting
        waiting_queue = self.waiting

        if waiting is not None:
            if scheduler_config.max_num_requests / 2 < len(
                    waiting_queue) < scheduler_config.max_num_requests:
                time.sleep(waiting)

        scheduled_requests = []
        ignored_requests = []
        while waiting_queue:
            if budget.num_curr_request == budget.max_num_requests:
                break

            request = waiting_queue[0]
            if self.record_metrics:
                scheduled_ts = time.perf_counter()
            else:
                scheduled_ts = None

            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                waiting_queue.popleft()
                continue

            if not isinstance(request, SchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            request = cast(SchedulableRequest, request)
            request.set_scheduled_ts(scheduled_ts)

            num_new_tokens = request.num_new_tokens

            if num_new_tokens > scheduler_config.max_model_len:
                self.requests.remove(request.request_id)
                waiting_queue.popleft()
                ignored_requests.append(request)
                continue

            if not budget.can_schedule(num_new_tokens=num_new_tokens):
                break

            budget.add_num_batched_tokens(request.request_id, num_new_tokens)
            waiting_queue.popleft()

            scheduled_requests.append(request)

        if self.record_metrics:
            scheduling_end_ts = time.perf_counter()
            scheduling_time = scheduling_end_ts - scheduling_begin_ts
            num_requests = budget.num_curr_request
            num_batched_tokens = budget.num_batched_tokens
            for request in scheduled_requests:
                request.metrics.scheduling_time = scheduling_time
                request.metrics.num_requests = num_requests
                request.metrics.num_batched_tokens = num_batched_tokens

        return PrefillOnlySchedulerOutput(
            scheduled_requests=scheduled_requests,
            ignored_requests=ignored_requests)
