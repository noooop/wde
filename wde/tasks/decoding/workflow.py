from wde.tasks.core.workflow import Workflow


class DecodeOnlyDecodingWorkflow(Workflow):
    Tokenizer: str = "wde.tasks.decoding.processor.tokenizer:Tokenizer"
    EngineArgs: str = "wde.tasks.decoding.arg_utils:DecodingEngineArgs"
    InputProcessor: str = ("wde.tasks.core.processor."
                           "input_processor:TextInputProcessor")
    RequestProcessor: str = ("wde.tasks.decoding.processor.input_processor:"
                             "DecodingModelRequestProcessor")
    OutputProcessor: str = ("wde.tasks.decoding.processor.output_processor:"
                            "DecodingModelOutputProcessor")
    ModelInputBuilder: str = (
        "wde.tasks.decoding.processor.model_input_builder:"
        "DecodingModelPreProcessor")
    Worker: str = "wde.tasks.decoding.worker.gpu_worker:Worker"
    Executor: str = "wde.tasks.decoding.executor.gpu_executor"
    Scheduler: str = "wde.tasks.decoding.scheduler:DecodingScheduler"
    AttnBackend: str = ("wde.tasks.decoding.backends.attention.selector:"
                        "DecodingAttnBackend")
    attn_type: str = "DECODER"

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

        if engine.engine_config.scheduler_config.scheduling in ["sync"]:
            workflow.Executor += ":GPUExecutor"
        elif engine.engine_config.scheduler_config.scheduling in [
                "simple_async", "async", "double_buffer"
        ]:
            workflow.Executor += ":GPUAsyncExecutor"

        return workflow
