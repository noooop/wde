import atexit
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
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
        self.executor = FrierenExecutor(self.worker)

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


class FrierenExecutor:

    def __init__(self, worker: WorkerBase):
        self.stream_pool = Queue()
        self.worker = worker

    def get_stream(self):
        if self.stream_pool.empty():
            stream = torch.cuda.Stream()
            return stream
        else:
            return self.stream_pool.get()

    def put_stream(self, stream):
        return self.stream_pool.put(stream)

    def execute_model(self, execute_input: ExecuteInput) -> ExecuteOutput:
        stream = self.get_stream()

        with torch.cuda.stream(stream):
            self.worker.non_blocking_h2d(execute_input)
            execute_output = self.worker(execute_input)
            self.worker.non_blocking_d2h(execute_output)

        stream.synchronize()
        self.put_stream(stream)

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
        thread = ThreadPoolExecutor(1)

        # Is there a better way to do it asynchronously?
        def _put(stream, scheduler_output, execute_output):
            stream.synchronize()
            self.put_stream(stream)
            executor_out.put((scheduler_output, execute_output))

        try:
            while True:
                o = executor_in.get()
                if o is None:
                    break

                stream = self.get_stream()

                scheduler_output, execute_input = o

                with torch.cuda.stream(stream):
                    self.worker.non_blocking_h2d(execute_input)
                    execute_output = self.worker(execute_input)
                    self.worker.non_blocking_d2h(execute_output)

                thread.submit(_put, stream, scheduler_output, execute_output)
        except Exception as e:
            executor_out.put(e)
        thread.shutdown()

    def double_buffer_execute_loop(self, executor_in: Queue,
                                   executor_out: Queue):
        from wde.tasks.core.schema.engine_io import SchedulerOutput
        worker = self.worker
        thread = ThreadPoolExecutor(1)

        @dataclass
        class Task:
            scheduler_output: SchedulerOutput
            execute_input: ExecuteInput
            execute_output: Optional[ExecuteOutput]
            stream: torch.cuda.Stream()

            @classmethod
            def get(cls, block=True, timeout=None):
                o = executor_in.get(block, timeout)
                if o is None:
                    return None

                scheduler_output, execute_input = o

                task = cls(scheduler_output=scheduler_output,
                           execute_input=execute_input,
                           execute_output=None,
                           stream=self.get_stream())
                return task

        def _put(stream, scheduler_output, execute_output):
            stream.synchronize()
            self.put_stream(stream)
            executor_out.put((scheduler_output, execute_output))

        def _prefetch():
            # Is there any way to achieve
            # poller = epoll.register(compute_stream, executor_in)
            # poller.poll()

            try:
                task = Task.get(timeout=0.001)

                if task is None:
                    return False
                else:
                    with torch.cuda.stream(task.stream):
                        worker.non_blocking_h2d(task.execute_input)
                        task.execute_output = worker(task.execute_input)
                    return task
            except queue.Empty:
                return None

        current_task: Optional[Task] = None
        next_task: Optional[Task] = None

        go_on = True

        try:
            while go_on:
                if current_task is None:
                    current_task = Task.get(block=True)
                    if current_task is None:
                        break

                    with torch.cuda.stream(current_task.stream):
                        worker.non_blocking_h2d(current_task.execute_input)
                        current_task.execute_output = worker(
                            current_task.execute_input)

                with torch.cuda.stream(current_task.stream):
                    self.worker.non_blocking_d2h(current_task.execute_output)

                f = thread.submit(_prefetch)
                thread.submit(_put, current_task.stream,
                              current_task.scheduler_output,
                              current_task.execute_output)

                maybe_task = f.result()
                if maybe_task is False:
                    go_on = False
                elif isinstance(maybe_task, Task):
                    next_task = maybe_task
                else:
                    next_task = None

                # switch double buffer
                current_task = next_task
                next_task = None
        except Exception as e:
            executor_out.put(e)
        thread.shutdown()
