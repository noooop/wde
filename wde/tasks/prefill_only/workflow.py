from wde.tasks.core.workflow import Workflow


class PrefillOnlyWorkflow(Workflow):
    InputProcessor: str = ("wde.tasks.core.processor."
                           "input_processor:TextInputProcessor")
    RequestProcessor: str = ("wde.tasks.core.processor."
                             "input_processor:TextRequestProcessor")
    ModelInputBuilder: str = (
        "wde.tasks.prefill_only.processor."
        "model_input_builder:PrefillOnlyModelInputBuilder")
    Worker: str = "wde.tasks.prefill_only.worker.gpu_worker:Worker"
    Executor: str = "wde.tasks.prefill_only.executor.gpu_executor"
    Scheduler: str = ("wde.tasks.prefill_only.scheduler:"
                      "PrefillOnlyScheduler")
    AttnBackend: str = ("wde.tasks.prefill_only.backends."
                        "attention.selector:AttnBackend")
    Tokenizer: str = "wde.tasks.prefill_only.processor.tokenizer:Tokenizer"

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
                "wde.tasks.prefill_only.executor.gpu_data_parallelism_executor:"
                "GPUDataParallelismExecutor")

        return workflow
