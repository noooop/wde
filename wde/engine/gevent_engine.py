# ruff: noqa: E402
import gevent
from gevent.monkey import patch_time

patch_time()

import time
from typing import Dict, Iterator, Optional, Sequence, Union

from gevent import Greenlet
from gevent.event import Event
from gevent.queue import Queue

from wde.logger import init_logger
from wde.tasks.core.llm_engine import LLMEngine
from wde.tasks.core.schema.engine_io import (Inputs, Params, PromptInputs,
                                             RequestOutput)
from wde.tasks.reranker.schema.engine_io import RerankerInputs

logger = init_logger(__name__)


class GeventEngineDeadError(RuntimeError):
    pass


class GeventStream:

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self._queue: Queue = Queue()
        self._finished = False

    def put(self, item: Union[RequestOutput, Exception]) -> None:
        if self._finished:
            return
        self._queue.put(item)

    def finish(self) -> None:
        self._queue.put(StopIteration())
        self._finished = True

    @property
    def finished(self) -> bool:
        return self._finished

    def __iter__(self):
        return self

    def __next__(self) -> RequestOutput:
        result = self._queue.get()
        if isinstance(result, Exception):
            raise result
        return result


class GeventLLMEngine:

    def __init__(
        self,
        model: str,
        dtype: str = "auto",
        seed: int = 0,
        **kwargs,
    ) -> None:
        engine_args = dict(
            model=model,
            dtype=dtype,
            seed=seed,
            **kwargs,
        )
        self.engine = LLMEngine.from_engine_args(engine_args)

        assert self.engine.use_async_scheduling

        self.background_loop = None
        self._new_requests_event = Event()
        self._request_streams: Dict[str, GeventStream] = {}

    @property
    def info(self):
        return {"mode": self.engine.engine_config.model_config.model}

    @property
    def is_running(self) -> bool:
        return (self.background_loop is not None
                and not self.background_loop.dead)

    @property
    def is_stopped(self) -> bool:
        return not self.is_running

    def ensure_start_execute_loop(self):
        if self.background_loop is None:
            self.background_loop = Greenlet.spawn(self.run_engine_loop)

    def run_engine_loop(self):
        while True:
            if not self.engine.has_unfinished_requests():
                self._new_requests_event.wait()

            while self.engine.has_unfinished_requests():
                # Not thread safe
                step_outputs = self.engine.step()

                for output in step_outputs:
                    request_id = output.request_id
                    self._request_streams[request_id].put(output)
                    if output.finished:
                        self._request_streams[request_id].finish()
                gevent.idle()
            self._new_requests_event.clear()

    def add_request(
        self,
        request_id: str,
        inputs: PromptInputs,
        params: Params,
        arrival_time: Optional[float] = None,
    ):
        self.ensure_start_execute_loop()
        self.engine.add_request(request_id=request_id,
                                inputs=inputs,
                                params=params,
                                arrival_time=arrival_time)

        stream = GeventStream(request_id)
        self._request_streams[request_id] = stream
        self._new_requests_event.set()
        return stream

    def encode(
        self,
        request_id: str,
        inputs: PromptInputs,
        pooling_params: Optional[Params] = None,
    ) -> Iterator[RequestOutput]:
        return self._process_request(
            request_id,
            inputs,
            pooling_params,
        )

    def compute_score(
        self,
        request_id: str,
        inputs: RerankerInputs,
        params: Optional[Union[Params, Sequence[Params]]] = None,
    ) -> Iterator[RequestOutput]:
        return self._process_request(
            request_id,
            inputs,
            params,
        )

    def _process_request(
        self,
        request_id: str,
        inputs: Inputs,
        params: Optional[Union[Params, Sequence[Params]]],
    ) -> Iterator[Union[RequestOutput]]:
        """Common logic to process requests with SamplingParams or
        PoolingParams."""
        arrival_time = time.perf_counter()

        stream = self.add_request(
            request_id,
            inputs,
            params,
            arrival_time=arrival_time,
        )

        try:
            for request_output in stream:
                yield request_output
        except (Exception, ) as e:
            self._abort(request_id)
            del self._request_streams[request_id]
            raise e

        del self._request_streams[request_id]

    def abort(self, request_id: str) -> None:
        if not self.is_running:
            raise GeventEngineDeadError(
                "Background loop is not running. If it was running, "
                "inspect the output to find the stacktrace of the "
                "error that caused the background loop to stop "
                "(GeventEngineDeadError).")

        return self._abort(request_id)

    def _abort(self, request_id: str) -> None:
        self.engine.abort_request(request_id)

    def terminate(self):
        self.background_loop.kill()
        self.engine = None
