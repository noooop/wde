import time
from dataclasses import dataclass, field
from typing import Set, cast

from wde.logger import init_logger
from wde.tasks.core.processor.input_processor import RequestProcessor
from wde.tasks.core.scheduler import Scheduler
from wde.tasks.prefill_only.config import PrefillOnlySchedulerConfig
from wde.tasks.prefill_only.schema.engine_io import (
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
        scheduler_config: PrefillOnlySchedulerConfig,
        request_processor: RequestProcessor,
    ) -> None:
        super().__init__(scheduler_config, request_processor)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.request_processor)

    def schedule(self) -> PrefillOnlySchedulerOutput:
        scheduling_begin_ts = time.perf_counter()

        budget = PrefillOnlySchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_seqs,
        )

        waiting = self.scheduler_config.waiting
        waiting_queue = self.waiting

        if waiting is not None:
            if len(waiting_queue) < self.scheduler_config.max_num_seqs:
                time.sleep(waiting)

        scheduled_requests = []
        ignored_requests = []
        while waiting_queue:
            request = waiting_queue[0]

            if request.request_id in self.aborted_requests:
                self.aborted_requests.remove(request.request_id)
                waiting_queue.popleft()
                continue

            if not isinstance(request, SchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            request = cast(SchedulableRequest, request)

            num_new_tokens = request.num_new_tokens

            if num_new_tokens > self.scheduler_config.max_model_len:
                self.requests.remove(request.request_id)
                waiting_queue.popleft()
                ignored_requests.append(request)
                continue

            if not budget.can_schedule(num_new_tokens=num_new_tokens):
                break

            budget.add_num_batched_tokens(request.request_id, num_new_tokens)
            waiting_queue.popleft()

            request.metrics.first_scheduled_ts = time.perf_counter()
            request.metrics.waiting_time = request.metrics.first_scheduled_ts - request.arrival_time
            scheduled_requests.append(request)

        scheduling_end_ts = time.perf_counter()
        scheduler_time = scheduling_end_ts - scheduling_begin_ts
        n_request_in_batch = len(scheduled_requests)
        for request in scheduled_requests:
            request.metrics.scheduler_time = scheduler_time
            request.metrics.n_request_in_batch = n_request_in_batch
            request.metrics.scheduling_end_ts = scheduling_end_ts

        return PrefillOnlySchedulerOutput(
            scheduled_requests=scheduled_requests,
            ignored_requests=ignored_requests)
