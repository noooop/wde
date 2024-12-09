from wde.workflows.core.workflow import Workflow


class PrefillOnlyWorkflow(Workflow):
    InputProcessor: str = ("wde.workflows.core.processor."
                           "input_processor:TextInputProcessor")
    RequestProcessor: str = ("wde.workflows.core.processor."
                             "input_processor:TextRequestProcessor")
    ModelInputBuilder: str = (
        "wde.workflows.prefill_only.processor."
        "model_input_builder:PrefillOnlyModelInputBuilder")

    Scheduler: str = ("wde.workflows.prefill_only.scheduler:"
                      "PrefillOnlyScheduler")
    AttnBackend: str = ("wde.workflows.prefill_only.backends."
                        "attention.selector:AttnBackend")
    Tokenizer: str = "wde.workflows.prefill_only.processor.tokenizer:Tokenizer"

    Executor: str = "wde.workflows.core.executor.gpu_executor"
    Worker: str = "wde.workflows.core.worker.gpu_worker:GPUWorker"
    Runer: str = "wde.workflows.prefill_only.runner.gpu_runner:PrefillOnlyGPURunner"

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

        if engine.engine_config.parallel_config is None:
            if engine.engine_config.scheduler_config.scheduling in ["sync"]:
                workflow.Executor += ":GPUExecutor"
            elif engine.engine_config.scheduler_config.scheduling in [
                    "simple_async", "async", "double_buffer"
            ]:
                workflow.Executor += ":GPUAsyncExecutor"
        else:
            assert engine.engine_config.parallel_config.data_parallel_size > 0
            assert engine.engine_config.scheduler_config.scheduling in [
                "async", "double_buffer"
            ]

            engine.engine_config.scheduler_config.max_num_on_the_fly *= (
                engine.engine_config.parallel_config.data_parallel_size)

            workflow.Executor = (
                "wde.workflows.prefill_only.executor.gpu_data_parallelism_executor:"
                "GPUDataParallelismExecutor")

        return workflow
