from dataclasses import dataclass, field

import pytest

from wde.workflows.core.config import SYSConfig
from wde.workflows.core.processor.input_processor import (Request,
                                                          RequestProcessor)
from wde.workflows.core.schema.engine_io import (RequestOutput, TextOnlyInputs,
                                                 TextSchedulableRequest)
from wde.workflows.prefill_only.config import \
    PrefillOnlySchedulerConfig as SchedulerConfig
from wde.workflows.prefill_only.scheduler import PrefillOnlyScheduler


class RequestProcessor4Test(RequestProcessor):

    def __init__(self, num_new_tokens):
        self.num_new_tokens = num_new_tokens

    def __call__(self, request: Request) -> TextSchedulableRequest:
        schedulable_request = TextSchedulableRequest(
            request_id=request.request_id,
            inputs=TextOnlyInputs(prompt_token_ids=[0] * self.num_new_tokens),
            arrival_time=request.arrival_time)
        return schedulable_request

    def from_engine(cls, engine):
        pass


@dataclass
class EngineConfig4Test:
    scheduler_config: SchedulerConfig
    sys_config: SYSConfig = field(default_factory=SYSConfig)


@pytest.mark.parametrize("num_new_tokens", [9, 99, 199])
@pytest.mark.parametrize("n_request", [9, 99, 199])
@pytest.mark.parametrize("max_num_requests", [1, 2, 3, 5, 7])
def test_limited_by_max_num_requests(n_request: int, num_new_tokens: int,
                                     max_num_requests: int):
    max_model_len = num_new_tokens + 1

    engine_config = EngineConfig4Test(
        scheduler_config=SchedulerConfig(max_num_batched_tokens=max_model_len *
                                         max_num_requests,
                                         max_model_len=max_model_len,
                                         max_num_requests=max_num_requests,
                                         frieren_executor_max_workers=2))

    scheduler = PrefillOnlyScheduler(
        engine_config=engine_config,
        request_processor=RequestProcessor4Test(num_new_tokens=num_new_tokens))

    for i in range(1, n_request + 1):
        scheduler.add_request(Request(request_id=str(i), arrival_time=0.))

    while scheduler.has_unfinished_requests():
        scheduler_output = scheduler.schedule()

        request_outputs = [
            RequestOutput(request_id=request.request_id, finished=True)
            for request in scheduler_output.scheduled_requests
        ]

        scheduler.free_finished_request(request_outputs)

        if scheduler.has_unfinished_requests():
            assert len(scheduler_output.scheduled_requests) == max_num_requests
        else:
            assert len(scheduler_output.scheduled_requests) <= max_num_requests
        assert len(scheduler_output.ignored_requests) == 0


@pytest.mark.parametrize("num_new_tokens", [9, 99, 199])
@pytest.mark.parametrize("n_request", [9, 99, 199])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
def test_limited_by_token_budget(n_request: int, num_new_tokens: int,
                                 max_num_requests: int):
    engine_config = EngineConfig4Test(scheduler_config=SchedulerConfig(
        max_model_len=num_new_tokens + 1,
        max_num_requests=max_num_requests,
        max_num_batched_tokens=(num_new_tokens + 1) * (max_num_requests - 1)))

    scheduler = PrefillOnlyScheduler(
        engine_config=engine_config,
        request_processor=RequestProcessor4Test(num_new_tokens=num_new_tokens))

    for i in range(1, n_request + 1):
        scheduler.add_request(Request(request_id=str(i), arrival_time=0.))

    n_scheduled_requests = 0
    while scheduler.has_unfinished_requests():
        scheduler_output = scheduler.schedule()
        n_scheduled_requests += len(scheduler_output.scheduled_requests)

        request_outputs = [
            RequestOutput(request_id=request.request_id, finished=True)
            for request in scheduler_output.scheduled_requests
        ]

        scheduler.free_finished_request(request_outputs)

        if scheduler.has_unfinished_requests():
            assert len(
                scheduler_output.scheduled_requests) == max_num_requests - 1
        else:
            assert len(
                scheduler_output.scheduled_requests) <= max_num_requests - 1
        assert len(scheduler_output.ignored_requests) == 0

    assert n_scheduled_requests == n_request


@pytest.mark.parametrize("num_new_tokens", [9, 99, 199])
@pytest.mark.parametrize("n_request", [9, 99, 199])
@pytest.mark.parametrize("max_num_requests", [2, 3, 5, 7])
def test_ignored_requests(n_request: int, num_new_tokens: int,
                          max_num_requests: int):
    max_model_len = num_new_tokens // 2

    engine_config = EngineConfig4Test(
        scheduler_config=SchedulerConfig(max_num_batched_tokens=max_model_len *
                                         max_num_requests,
                                         max_model_len=max_model_len,
                                         max_num_requests=max_num_requests))

    scheduler = PrefillOnlyScheduler(
        engine_config=engine_config,
        request_processor=RequestProcessor4Test(num_new_tokens=num_new_tokens))

    for i in range(1, n_request + 1):
        scheduler.add_request(Request(request_id=str(i), arrival_time=0.))

    n_ignored_requests = 0
    while scheduler.has_unfinished_requests():
        scheduler_output = scheduler.schedule()

        assert len(scheduler_output.scheduled_requests) == 0
        assert len(scheduler_output.ignored_requests) > 0

        n_ignored_requests += len(scheduler_output.ignored_requests)

    assert n_ignored_requests == n_request
