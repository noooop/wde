from wde.tasks.chat.schema.api import PROTOCOL
from wde.workflows.core.workflow import Workflow


class DecodeOnlyDecodingWorkflow(Workflow):
    Tokenizer: str = "wde.workflows.decoding.backends.sampling.detokenizer:Tokenizer"
    EngineArgs: str = "wde.workflows.decoding.arg_utils:DecodingEngineArgs"
    InputProcessor: str = ("wde.workflows.core.processor."
                           "input_processor:TextInputProcessor")
    RequestProcessor: str = (
        "wde.workflows.decoding.processor.input_processor:"
        "DecodingModelRequestProcessor")
    OutputProcessor: str = (
        "wde.workflows.decoding.processor.output_processor:"
        "DecodingModelOutputProcessor")
    ModelInputBuilder: str = (
        "wde.workflows.decoding.processor.model_input_builder:"
        "DecodingModelInputBuilder")
    Worker: str = "wde.workflows.decoding.worker.gpu_worker:Worker"
    Executor: str = "wde.workflows.decoding.executor.gpu_executor"
    Scheduler: str = "wde.workflows.decoding.scheduler:DecodingScheduler"
    AttnBackend: str = ("wde.workflows.decoding.backends.attention.selector:"
                        "DecodingAttnBackend")
    attn_type: str = "DECODER"
    protocol: str = PROTOCOL

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
