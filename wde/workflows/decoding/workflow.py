from wde.tasks.chat.schema.api import PROTOCOL
from wde.workflows.core.workflow import Workflow
from wde.workflows.decoding.scheduler import get_scheduler_and_block_allocator


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

    AttnBackend: str = ("wde.workflows.decoding.backends.attention.selector:"
                        "DecodingAttnBackend")

    Executor: str = "wde.workflows.core.executor.gpu_executor"
    Worker: str = "wde.workflows.core.worker.gpu_worker:GPUWorker"
    Runer: str = "wde.workflows.decoding.runner.gpu_runner:GPUDecodingRunner"

    KVCacheManager: str = "wde.workflows.decoding.kv_cache.physical_manager:PhysicalGPUKVCacheManager"

    attn_type: str = "DECODER"
    protocol: str = PROTOCOL

    Scheduler: str
    BlockAllocator: str

    @classmethod
    def from_engine(cls, engine):
        workflow = cls()

        if engine.engine_config.scheduler_config.scheduling in ["sync"]:
            workflow.Executor += ":GPUExecutor"
        elif engine.engine_config.scheduler_config.scheduling in [
                "simple_async", "async"
        ]:
            workflow.Executor += ":GPUAsyncExecutor"

        Scheduler, BlockAllocator = get_scheduler_and_block_allocator(
            engine.engine_config)

        workflow.Scheduler = Scheduler
        workflow.BlockAllocator = BlockAllocator

        return workflow
