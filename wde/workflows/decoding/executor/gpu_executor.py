import atexit
from queue import Queue
from typing import Optional, Tuple

from wde.logger import init_logger
from wde.workflows.core.config import EngineConfig
from wde.workflows.core.executor.gpu_executor import FrierenExecutor
from wde.workflows.core.llm_engine import LLMEngine
from wde.workflows.core.schema.execute_io import ExecuteInput, ExecuteOutput
from wde.workflows.core.worker import WorkerWrapperBase
from wde.workflows.core.workflow import Workflow
from wde.workflows.decoding.backends.attention import \
    DecodeOnlyAttentionBackend

logger = init_logger(__name__)


def create_worker(module, **kwargs):
    module_name, class_name = module.split(":")
    wrapper = WorkerWrapperBase(
        worker_module_name=module_name,
        worker_class_name=class_name,
    )
    wrapper.init_worker(**kwargs)
    return wrapper.worker


class GPUExecutor:
    support_scheduling = ["sync_scheduling"]

    def __init__(
        self,
        engine_config: EngineConfig,
        workflow: Workflow,
        attn_backend: DecodeOnlyAttentionBackend,
    ) -> None:
        self.engine_config = engine_config
        self.workflow = workflow
        self.attn_backend = attn_backend
        self._init_executor()
        self.executor = FrierenExecutor.from_engine_config(
            engine_config=self.engine_config,
            worker=self.worker,
        )

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

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks by invoking the
        underlying worker.
        """
        return self.worker.determine_num_available_blocks()

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks) -> None:
        """Initialize the KV cache by invoking the underlying worker.
        """
        # NOTE: This is logged in the executor because there can be >1 worker
        # with other executors. We could log in the engine level, but work
        # remains to abstract away the device for non-GPU configurations.
        logger.info("# GPU blocks: %d, # CPU blocks: %d", num_gpu_blocks,
                    num_cpu_blocks)

        self.worker.initialize_cache(num_gpu_blocks, num_cpu_blocks)

    def execute_model(self,
                      execute_input: ExecuteInput) -> Optional[ExecuteOutput]:
        return self.executor.execute_model(execute_input)

    def initialize_kv_caches(self, engine: LLMEngine) -> None:
        """Initialize the KV cache in the worker(s).

        The workers will determine the number of blocks in both the GPU cache
        and the swap CPU cache.
        """
        self.worker.model_runner.prepare_model_input = engine.model_inputs_builder.prepare_model_input

        num_gpu_blocks, num_cpu_blocks = (
            self.determine_num_available_blocks())

        if self.engine_config.cache_config.num_gpu_blocks_override is not None:
            num_gpu_blocks_override = self.engine_config.cache_config.num_gpu_blocks_override
            logger.info(
                "Overriding num_gpu_blocks=%d with "
                "num_gpu_blocks_override=%d", num_gpu_blocks,
                num_gpu_blocks_override)
            num_gpu_blocks = num_gpu_blocks_override

        self.engine_config.cache_config.num_gpu_blocks = num_gpu_blocks
        self.engine_config.cache_config.num_cpu_blocks = num_cpu_blocks

        self.initialize_cache(num_gpu_blocks, num_cpu_blocks)

        del self.worker.model_runner.prepare_model_input

    def shutdown_execute_loop(self):
        pass


class GPUAsyncExecutor(GPUExecutor):
    support_scheduling = ["async_scheduling"]

    def __init__(self, engine_config: EngineConfig, workflow: Workflow,
                 attn_backend: DecodeOnlyAttentionBackend, executor_in: Queue,
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
