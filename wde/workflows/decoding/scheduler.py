import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Set, Tuple, cast

from wde.logger import init_logger
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.core.scheduler import Scheduler
from wde.workflows.core.schema.engine_io import RequestOutput
from wde.workflows.decoding.kv_cache.manager import AllocStatus, KVCacheManager
from wde.workflows.decoding.schema.engine_io import (
    DecodingSchedulableRequest, DecodingSchedulerOutput)
from wde.workflows.decoding.schema.request import RequestStatus

logger = init_logger(__name__)


@dataclass
class DecodingSchedulingBudget:
    token_budget: int
    max_num_requests: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_requests: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_requests: int):
        assert num_new_tokens != 0
        assert num_new_requests != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_requests + num_new_requests
                <= self.max_num_requests)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_requests(self, req_id: str, num_curr_requests: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_requests += num_curr_requests

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_requests -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_requests(self):
        return self._num_curr_requests

    def full(self) -> bool:
        if self.num_batched_tokens >= self.token_budget:
            return True

        if self.num_curr_requests >= self.max_num_requests:
            return True

        return False


@dataclass
class SchedulerRunningOutputs:
    decode_requests: List[DecodingSchedulableRequest]
    prefill_requests: List[DecodingSchedulableRequest]
    preempted: List[DecodingSchedulableRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(decode_requests=[],
                                       prefill_requests=[],
                                       preempted=[])


@dataclass
class SchedulerPrefillOutputs:
    scheduled_requests: List[DecodingSchedulableRequest]
    ignored_requests: List[DecodingSchedulableRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            scheduled_requests=[],
            ignored_requests=[],
        )


class DecodingScheduler(Scheduler):
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(self, engine_config: EngineConfig,
                 request_processor: RequestProcessor,
                 kv_cache_manager: KVCacheManager) -> None:
        super().__init__(engine_config, request_processor)
        self.running: Deque[DecodingSchedulableRequest] = deque()
        self.kv_cache_manager = kv_cache_manager
        self.record_metrics = engine_config.sys_config.record_metrics
        self.num_cumulative_preemption = 0

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config, engine.request_processor,
                   engine.kv_cache_manager)

    def _schedule_prefills(
        self,
        budget: DecodingSchedulingBudget,
    ) -> SchedulerPrefillOutputs:
        ignored_requests: List[DecodingSchedulableRequest] = []
        scheduled_requests: List[DecodingSchedulableRequest] = []

        waiting_queue = self.waiting
        while waiting_queue:
            if budget.full():
                break

            request = waiting_queue[0]

            if self.record_metrics:
                scheduled_ts = time.perf_counter()

            # 1. Check if it has been aborted
            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                waiting_queue.popleft()
                continue

            # 2. Check if request been preprocessored
            if not isinstance(request, DecodingSchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            request = cast(DecodingSchedulableRequest, request)

            # 3. Check if the input exceeds the maximum length
            num_new_tokens = request.get_num_new_tokens()
            prompt_limit = self.scheduler_config.max_model_len
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)

                ignored_requests.append(request)
                waiting_queue.popleft()
                continue

            # 4. Check if kv cache can be allocated
            can_allocate = self.kv_cache_manager.can_allocate(request)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of kv_cache_manager",
                    num_new_tokens)
                request.status = RequestStatus.FINISHED_IGNORED
                ignored_requests.append(request)
                waiting_queue.popleft()
                continue

            # 5. Allocate kv cache
            waiting_queue.popleft()
            self._allocate_and_set_running(request)

            # 6. chunked prefill
            budget_bound_token_chunk_size = min(
                num_new_tokens, budget.remaining_token_budget())

            if budget_bound_token_chunk_size == 0:
                # No budget => Stop
                break

            # Can schedule this request.
            request.token_chunk_size = budget_bound_token_chunk_size
            scheduled_requests.append(request)

            budget.add_num_batched_tokens(request.request_id,
                                          request.token_chunk_size)
            budget.add_num_requests(request.request_id, 1)

            if self.record_metrics:
                request.set_scheduled_ts(scheduled_ts)

        return SchedulerPrefillOutputs(scheduled_requests=scheduled_requests,
                                       ignored_requests=ignored_requests)

    def _schedule_running(self, budget: DecodingSchedulingBudget,
                          running_queue) -> SchedulerRunningOutputs:

        prefill_requests = []
        decode_requests = []

        blocks_to_copy = []
        preempted = []

        while running_queue:
            if budget.full():
                break

            request = running_queue[0]
            if self.record_metrics:
                scheduled_ts = time.perf_counter()

            num_new_tokens = request.get_num_new_tokens()
            assert num_new_tokens > 0

            # 1. chunked prefill
            budget_bound_token_chunk_size = min(
                num_new_tokens, budget.remaining_token_budget())

            if budget_bound_token_chunk_size == 0:
                # No budget => Stop
                break

            # 2. Check if kv cache can be allocated
            while not self._can_append_slots(request):
                if running_queue:
                    # Preempt the lowest-priority request.
                    victim_request = running_queue.pop()
                    self._preempt(victim_request)
                    preempted.append(victim_request)
                else:
                    # No other request can be preempted.
                    # Preempt the current request.
                    self._preempt(request)
                    preempted.append(request)
                    break
            else:
                self._append_slots(request, blocks_to_copy)

            running_queue.popleft()

            if self.record_metrics:
                request.set_scheduled_ts(scheduled_ts)

            if request.is_prefill:
                request.token_chunk_size = budget_bound_token_chunk_size
                prefill_requests.append(request)
            else:
                request.token_chunk_size = 1
                decode_requests.append(request)

            budget.add_num_batched_tokens(request.request_id,
                                          request.token_chunk_size)
            budget.add_num_requests(request.request_id, 1)

        return SchedulerRunningOutputs(decode_requests=decode_requests,
                                       prefill_requests=prefill_requests,
                                       preempted=preempted)

    def _schedule(self) -> DecodingSchedulerOutput:
        budget = DecodingSchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        scheduled_requests = []

        if self.running:
            running_queue, busy_requests = self._split_running_queue()

            if running_queue:
                running_scheduled = self._schedule_running(
                    budget, running_queue)

                self.running = running_queue

                self.running.extend(running_scheduled.decode_requests)
                self.running.extend(running_scheduled.prefill_requests)

                self.running.extend(busy_requests)

                self.waiting.extend(running_scheduled.preempted)

                scheduled_requests.extend(running_scheduled.decode_requests)
                scheduled_requests.extend(running_scheduled.prefill_requests)

            else:
                self.running = deque(busy_requests)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_requests <= self.scheduler_config.max_num_requests

        prefills = self._schedule_prefills(budget)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_requests <= self.scheduler_config.max_num_requests

        self.running.extend(prefills.scheduled_requests)
        scheduled_requests.extend(prefills.scheduled_requests)

        return DecodingSchedulerOutput(
            scheduled_requests=scheduled_requests,
            ignored_requests=prefills.ignored_requests,
            num_batched_tokens=budget.num_batched_tokens,
            num_requests=budget.num_curr_requests,
        )

    def schedule(self) -> Optional[DecodingSchedulerOutput]:
        if self.record_metrics:
            scheduling_begin_ts = time.perf_counter()

        scheduler_outputs = self._schedule()

        for request in scheduler_outputs.scheduled_requests:
            request.busy = True

        if self.record_metrics:
            scheduling_end_ts = time.perf_counter()
            scheduling_time = scheduling_end_ts - scheduling_begin_ts
            num_requests = scheduler_outputs.num_requests
            num_batched_tokens = scheduler_outputs.num_batched_tokens
            for request in scheduler_outputs.scheduled_requests:
                request.metrics.scheduling_time = scheduling_time
                request.metrics.num_requests = num_requests
                request.metrics.num_batched_tokens = num_batched_tokens

        return scheduler_outputs

    def free_finished_request(self,
                              request_outputs: List[RequestOutput]) -> None:

        request_ids = set(request.request_id for request in request_outputs)

        remaining: Deque[DecodingSchedulableRequest] = deque()
        for request in self.running:
            if request.finished:
                self.requests.remove(request.request_id)
                self.kv_cache_manager.free(request)
            else:
                remaining.append(request)
                if request.request_id in request_ids:
                    request.busy = False

        self.running = remaining

    def _allocate_and_set_running(self,
                                  request: DecodingSchedulableRequest) -> None:
        self.kv_cache_manager.allocate(request)
        request.status = RequestStatus.RUNNING

    def _split_running_queue(self):
        busy_requests = []
        running_queue = []
        for request in self.running:
            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                continue

            if request.request_id not in self.requests:
                # aborted_requests
                continue

            if request.busy:
                busy_requests.append(request)
            else:
                running_queue.append(request)

        running_queue: Deque[DecodingSchedulableRequest] = deque(
            sorted(running_queue, key=lambda request: request.arrival_time))

        return running_queue, busy_requests

    def _can_append_slots(self, request: DecodingSchedulableRequest) -> bool:
        return self.kv_cache_manager.can_append_slots(
            request=request,
            num_lookahead_slots=0,
        )

    def _append_slots(
        self,
        request: DecodingSchedulableRequest,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        cows = self.kv_cache_manager.append_slots(request, 0)
        assert not cows
        blocks_to_copy.extend(cows)

    def _preempt(
        self,
        request: DecodingSchedulableRequest,
    ):
        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "request %s is preempted by RECOMPUTE mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", request.request_id,
                self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        request.status = RequestStatus.WAITING

        self.kv_cache_manager.free(request)
        request.reset_state_for_recompute()
