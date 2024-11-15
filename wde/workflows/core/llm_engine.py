import time
from queue import Empty, Queue
from typing import Dict, Iterable, List, Optional, Type, Union

from wde.logger import init_logger
from wde.workflows.core.arg_utils import EngineArgs
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.schema.engine_io import Inputs, Params, RequestOutput
from wde.workflows.core.workflow import Workflow

logger = init_logger(__name__)


def lazy_import(module):
    module_name, class_name = module.split(":")
    import importlib
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


class LLMEngine:

    def __init__(self, engine_config: EngineConfig,
                 workflow_cls: Type[Workflow]) -> None:
        self.engine_config = engine_config
        self.engine_config.log_config()
        self.workflow = workflow_cls.from_engine(self)

        self._maybe_init_async_scheduling()

        self.attn_backend = lazy_import(
            self.workflow.AttnBackend).from_engine(self)
        self.executor = lazy_import(self.workflow.Executor).from_engine(self)
        self.tokenizer = lazy_import(self.workflow.Tokenizer).from_engine(self)
        self.model_inputs_builder = lazy_import(
            self.workflow.ModelInputBuilder).from_engine(self)

        if hasattr(self.executor, "initialize_kv_caches"):
            self.executor.initialize_kv_caches(self)

        self.input_processor = lazy_import(
            self.workflow.InputProcessor).from_engine(self)
        self.request_processor = lazy_import(
            self.workflow.RequestProcessor).from_engine(self)
        self.scheduler = lazy_import(self.workflow.Scheduler).from_engine(self)
        self.output_processor = lazy_import(
            self.workflow.OutputProcessor).from_engine(self)

    def _maybe_init_async_scheduling(self):
        executor_cls = lazy_import(self.workflow.Executor)
        scheduler_cls = lazy_import(self.workflow.Scheduler)

        if ("async_scheduling" in executor_cls.support_scheduling
                and "async_scheduling" in scheduler_cls.support_scheduling):
            logger.info("Use async scheduling")
            self.use_async_scheduling = True

        elif ("sync_scheduling" in executor_cls.support_scheduling
              and "sync_scheduling" in scheduler_cls.support_scheduling):
            logger.info("Use sync scheduling")
            self.use_async_scheduling = False

        else:
            raise RuntimeError(f"Executor support scheduling: "
                               f"{executor_cls.support_scheduling}."
                               f"Scheduler support scheduling: "
                               f"{scheduler_cls.support_scheduling}."
                               f"Not compatible")

        if self.use_async_scheduling:
            self.executor_in = Queue()
            self.executor_out = Queue()

            self.max_num_on_the_fly = (
                self.engine_config.scheduler_config.max_num_on_the_fly)
            self.num_on_the_fly = 0
            self.step = self.async_step
        else:
            self.step = self.sync_step

    @classmethod
    def from_engine_args(cls, engine_args: Union[Dict,
                                                 EngineArgs]) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        from wde.backends.loader.utils import get_model_workflow
        from wde.backends.models.transformers_utils.config import get_config

        if isinstance(engine_args, EngineArgs):
            engine_args = engine_args.to_dict()

        hf_config = get_config(engine_args["model"],
                               engine_args.get("trust_remote_code", False),
                               engine_args.get("revision", None),
                               engine_args.get("code_revision", None))

        workflow_cls_str = get_model_workflow(hf_config)
        workflow_cls = lazy_import(workflow_cls_str).from_engine_args(
            engine_args)
        engine_args = lazy_import(workflow_cls.EngineArgs)(**engine_args)

        engine_config = engine_args.create_engine_config()
        engine = cls(engine_config, workflow_cls)
        return engine

    def add_request(self,
                    request_id: str,
                    inputs: Optional[Union[str, Inputs]] = None,
                    params: Optional[Params] = None,
                    arrival_time: Optional[float] = None) -> None:
        request = self.input_processor(request_id, inputs, params,
                                       arrival_time)

        # The raised ValidationError will be passed to the upper call stack
        self.scheduler.add_request(request)

    def abort_request(self, request_id: Union[str, Iterable[str]]) -> None:
        self.scheduler.abort_request(request_id)

    def _record_inference_time(self, scheduler_output, executor_output):
        inference_time = executor_output.inference_end_ts - executor_output.inference_begin_ts
        for request in scheduler_output.scheduled_requests:
            request.metrics.inference_begin_ts = executor_output.inference_begin_ts
            request.metrics.inference_end_ts = executor_output.inference_end_ts
            request.metrics.inference_time = inference_time
            request.metrics.scheduling2inference = executor_output.inference_begin_ts - request.metrics.scheduled_ts

    def _record_latency(self, request_outputs):
        finish_ts = time.perf_counter()
        for request in request_outputs:
            request.metrics.finish_ts = finish_ts
            request.metrics.latency = finish_ts - request.metrics.scheduled_ts
            request.metrics.latency_so_far = finish_ts - request.metrics.first_scheduled_ts

    def sync_step(self) -> List[RequestOutput]:
        scheduler_output = self.scheduler.schedule()
        if scheduler_output.is_empty():
            return []

        executor_input = self.model_inputs_builder(scheduler_output)
        executor_output = self.executor.execute_model(executor_input)

        self._record_inference_time(scheduler_output, executor_output)

        request_outputs = self.output_processor(scheduler_output,
                                                executor_output)
        self.scheduler.free_finished_request(request_outputs)
        request_outputs = self.scheduler.remove_abort_request(request_outputs)

        self._record_latency(request_outputs)
        return request_outputs

    def async_step(self) -> List[RequestOutput]:
        self.executor.ensure_start_execute_loop()
        self._put_as_many_as_possible()

        if self.num_on_the_fly == 0:
            return []

        return self._get(block=True)

    def _put_as_many_as_possible(self):
        while self.num_on_the_fly < self.max_num_on_the_fly:
            scheduler_output = self.scheduler.schedule()

            if scheduler_output is None:
                return

            if scheduler_output.is_empty():
                break
            executor_input = self.model_inputs_builder(scheduler_output)

            self.executor_in.put((scheduler_output, executor_input))
            self.num_on_the_fly += 1

    def _get(self, block):
        try:
            maybe_except = self.executor_out.get(block)
        except Empty:
            return

        if isinstance(maybe_except, Exception):
            raise maybe_except

        scheduler_output, executor_output = maybe_except

        self.num_on_the_fly -= 1

        # Theoretically, this put is not needed
        # practically, task can be inqueue before doing post-processing
        self._put_as_many_as_possible()

        self._record_inference_time(scheduler_output, executor_output)

        request_outputs = self.output_processor(scheduler_output,
                                                executor_output)
        self.scheduler.free_finished_request(request_outputs)
        request_outputs = self.scheduler.remove_abort_request(request_outputs)

        self._record_latency(request_outputs)
        return request_outputs

    def get_num_unfinished_requests(self) -> int:
        """Gets the number of unfinished requests."""
        return self.scheduler.get_num_unfinished_requests()

    def has_unfinished_requests(self) -> bool:
        """Returns True if there are unfinished requests."""
        return self.scheduler.has_unfinished_requests()

    def __reduce__(self):
        # This is to ensure that the LLMEngine is not referenced in
        # the closure used to initialize Ray worker actors
        raise RuntimeError("LLMEngine should not be pickled!")

    def __del__(self):
        # Shutdown model executor when engine is garbage collected
        # Use getattr since __init__ can fail before the field is set
        if executor := getattr(self, "executor", None):
            executor.shutdown_execute_loop()
