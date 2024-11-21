import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Optional

import torch

from wde.workflows.core.schema.execute_io import ExecuteInput, ExecuteOutput
from wde.workflows.core.worker import WorkerBase


class FrierenExecutor:

    def __init__(self, worker: WorkerBase, max_workers=1):
        self.stream_pool = Queue()
        self.worker = worker
        self.max_workers = max_workers

    def get_stream(self):
        if self.stream_pool.empty():
            stream = torch.cuda.Stream()
            return stream
        else:
            return self.stream_pool.get()

    def put_stream(self, stream):
        return self.stream_pool.put(stream)

    def execute_model(self, execute_input: ExecuteInput) -> ExecuteOutput:
        inference_begin_ts = time.perf_counter()

        stream = self.get_stream()
        with torch.cuda.stream(stream):
            self.worker.non_blocking_h2d(execute_input)
            execute_output = self.worker(execute_input)
            self.worker.non_blocking_d2h(execute_output)

        stream.synchronize()
        self.put_stream(stream)

        inference_end_ts = time.perf_counter()

        execute_output.inference_begin_ts = inference_begin_ts
        execute_output.inference_end_ts = inference_end_ts
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
        thread = ThreadPoolExecutor(self.max_workers)

        # Is there a better way to do it asynchronously?
        def _put(stream, scheduler_output, execute_output):
            stream.synchronize()
            self.put_stream(stream)
            execute_output.inference_end_ts = time.perf_counter()
            executor_out.put((scheduler_output, execute_output))

        try:
            while True:
                o = executor_in.get()
                if o is None:
                    break

                inference_begin_ts = time.perf_counter()

                stream = self.get_stream()

                scheduler_output, execute_input = o

                with torch.cuda.stream(stream):
                    self.worker.non_blocking_h2d(execute_input)
                    execute_output = self.worker(execute_input)
                    self.worker.non_blocking_d2h(execute_output)

                execute_output.inference_begin_ts = inference_begin_ts
                thread.submit(_put, stream, scheduler_output, execute_output)
        except Exception as e:
            executor_out.put(e)
        thread.shutdown()

    def double_buffer_execute_loop(self, executor_in: Queue,
                                   executor_out: Queue):
        from wde.workflows.core.schema.engine_io import SchedulerOutput
        worker = self.worker
        thread = ThreadPoolExecutor(self.max_workers)

        @dataclass
        class Task:
            scheduler_output: SchedulerOutput
            execute_input: ExecuteInput
            execute_output: Optional[ExecuteOutput]
            stream: torch.cuda.Stream()
            inference_begin_ts: float

            @classmethod
            def get(cls, block=True, timeout=None):
                o = executor_in.get(block, timeout)
                if o is None:
                    return None

                scheduler_output, execute_input = o

                task = cls(scheduler_output=scheduler_output,
                           execute_input=execute_input,
                           execute_output=None,
                           stream=self.get_stream(),
                           inference_begin_ts=time.perf_counter())
                return task

        def _put(task):
            task.stream.synchronize()
            self.put_stream(task.stream)
            task.execute_output.inference_begin_ts = task.inference_begin_ts
            task.execute_output.inference_end_ts = time.perf_counter()
            executor_out.put((task.scheduler_output, task.execute_output))

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
                thread.submit(_put, current_task)

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
