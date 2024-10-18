import atexit
import queue
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

import torch

from wde.backends.attention import AttentionBackend
from wde.logger import init_logger
from wde.tasks.core.config import EngineConfig
from wde.tasks.core.llm_engine import LLMEngine
from wde.tasks.core.schema.execute_io import ExecuteInput, ExecuteOutput
from wde.tasks.core.worker import WorkerBase, create_worker
from wde.tasks.core.workflow import Workflow

logger = init_logger(__name__)


class GPUExecutor:
    support_scheduling = ["sync_scheduling"]

    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: AttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self._init_executor()
        self.executor = Executor(self.worker)

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend)

    def _init_executor(self) -> None:
        """Initialize the worker and load the model.
        """

        worker_kwargs = dict(
            engine_config=self.engine_config,
            attn_backend=self.attn_backend,
        )
        worker_kwargs.update(module=self.workflow.Worker)

        self.worker = create_worker(**worker_kwargs)
        self.worker.init_device()
        self.worker.load_model()

    def execute_model(self,
                      execute_input: ExecuteInput) -> Optional[ExecuteOutput]:
        return self.executor.execute_model(execute_input)

    def shutdown_execute_loop(self):
        pass


class GPUAsyncExecutor(GPUExecutor):
    support_scheduling = ["async_scheduling"]

    def __init__(self, engine_config: EngineConfig, workflow: Workflow,
                 attn_backend: AttentionBackend, executor_in: Queue,
                 executor_out: Queue) -> None:
        super().__init__(engine_config, workflow, attn_backend)
        from threading import Thread

        self.Thread = Thread
        self.executor_in = executor_in
        self.executor_out = executor_out

        self.executor_thread: Optional[Thread] = None

        if self.engine_config.scheduler_config.scheduling == "double_buffer":
            self.execute_loop = self.executor.double_buffer_execute_loop
        elif self.engine_config.scheduler_config.scheduling == "simple_async":
            self.execute_loop = self.executor.simple_async_execute_loop
        else:
            self.execute_loop = self.executor.async_execute_loop

    @classmethod
    def from_engine(cls, engine: LLMEngine):
        return cls(engine_config=engine.engine_config,
                   workflow=engine.workflow,
                   attn_backend=engine.attn_backend,
                   executor_in=engine.executor_in,
                   executor_out=engine.executor_out)

    def ensure_start_execute_loop(self):
        if self.executor_thread is None or not self.executor_thread.is_alive():
            self.executor_thread = self.Thread(target=self.execute_loop,
                                               args=(self.executor_in,
                                                     self.executor_out),
                                               daemon=True)
            self.executor_thread.start()
            atexit.register(self.shutdown_execute_loop)

    def shutdown_execute_loop(self):
        if self.executor_thread.is_alive():
            self.executor_in.put(None)
            self.executor_thread.join()
            atexit.unregister(self.shutdown_execute_loop)


class Executor:

    def __init__(self, worker: WorkerBase):
        self.h2d_stream = torch.cuda.Stream()
        self.compute_stream = torch.cuda.Stream()
        self.d2h_stream = torch.cuda.Stream()
        self.worker = worker

    def execute_model(self, execute_input: ExecuteInput) -> ExecuteOutput:
        with torch.cuda.stream(self.h2d_stream):
            self.worker.non_blocking_h2d(execute_input)

        self.compute_stream.wait_stream(self.h2d_stream)
        with torch.cuda.stream(self.compute_stream):
            execute_output = self.worker(execute_input)

        self.d2h_stream.wait_stream(self.compute_stream)
        with torch.cuda.stream(self.d2h_stream):
            self.worker.non_blocking_d2h(execute_output)

        self.d2h_stream.synchronize()

        return execute_output

    def simple_async_execute_loop(self, executor_in: Queue,
                                  executor_out: Queue):
        try:
            while True:
                o = executor_in.get()
                if o is None:
                    break

                scheduler_output, execute_input = o
                execute_output = self.execute_model(execute_input)
                executor_out.put((scheduler_output, execute_output))
        except Exception as e:
            executor_out.put(e)

    def async_execute_loop(self, executor_in: Queue, executor_out: Queue):
        put_thread = ThreadPoolExecutor(1)

        def _put(scheduler_output, execute_output):
            self.d2h_stream.wait_stream(self.compute_stream)
            with torch.cuda.stream(self.d2h_stream):
                self.worker.non_blocking_d2h(execute_output)
                self.d2h_stream.synchronize()
            executor_out.put((scheduler_output, execute_output))

        try:
            while True:
                o = executor_in.get()
                if o is None:
                    break

                scheduler_output, execute_input = o

                with torch.cuda.stream(self.h2d_stream):
                    self.worker.non_blocking_h2d(execute_input)

                self.compute_stream.wait_stream(self.h2d_stream)
                with torch.cuda.stream(self.compute_stream):
                    execute_output = self.worker(execute_input)

                put_thread.submit(_put, scheduler_output, execute_output)
        except Exception as e:
            executor_out.put(e)
        put_thread.shutdown()

    def double_buffer_execute_loop(self, executor_in: Queue,
                                   executor_out: Queue):

        from dataclasses import dataclass

        from wde.tasks.core.schema.engine_io import SchedulerOutput

        h2d_stream = self.h2d_stream
        compute_stream = self.compute_stream
        d2h_stream = self.d2h_stream
        worker = self.worker
        put_thread = ThreadPoolExecutor(1)

        # Is there a better way to do it asynchronously?
        def _put(scheduler_output, execute_output):
            d2h_stream.wait_stream(compute_stream)
            with torch.cuda.stream(d2h_stream):
                worker.non_blocking_d2h(execute_output)
                d2h_stream.synchronize()
            executor_out.put((scheduler_output, execute_output))

        @dataclass
        class Task:
            scheduler_output: SchedulerOutput
            execute_input: ExecuteInput
            execute_output: Optional[ExecuteOutput]

            @classmethod
            def get(cls, block=True, timeout=None):
                o = executor_in.get(block, timeout)
                if o is None:
                    return None

                scheduler_output, execute_input = o

                task = cls(scheduler_output=scheduler_output,
                           execute_input=execute_input,
                           execute_output=None)
                return task

        current_task: Optional[Task] = None
        next_task: Optional[Task] = None

        go_on = True

        try:
            while go_on:
                if current_task is None:
                    current_task = Task.get(block=True)
                    if current_task is None:
                        break

                    with torch.cuda.stream(h2d_stream):
                        worker.non_blocking_h2d(current_task.execute_input)

                    compute_stream.wait_stream(h2d_stream)
                    with torch.cuda.stream(compute_stream):
                        current_task.execute_output = worker(
                            current_task.execute_input)

                try:
                    # Is there any way to achieve
                    # poller = epoll.register(compute_stream, executor_in)
                    # poller.poll()
                    next_task = Task.get(timeout=0.002)
                    if next_task is None:
                        go_on = False
                    else:
                        with torch.cuda.stream(h2d_stream):
                            worker.non_blocking_h2d(next_task.execute_input)

                        compute_stream.wait_stream(h2d_stream)

                        with torch.cuda.stream(compute_stream):
                            next_task.execute_output = worker(
                                next_task.execute_input)
                except queue.Empty:
                    pass

                put_thread.submit(_put, current_task.scheduler_output,
                                  current_task.execute_output)
                current_task = next_task
                next_task = None
        except Exception as e:
            executor_out.put(e)
        put_thread.shutdown()
