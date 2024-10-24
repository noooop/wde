from wde.tasks.decode_only.workflow import DecodeOnlyWorkflow
from wde.tasks.encode_only.workflow import EncodeOnlyWorkflow
from wde.tasks.retriever.engine.schema import PROTOCOL


class RetrieverEncodeOnlyWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = ("wde.tasks.retriever.processor."
                            "output_processor:RetrieverOutputProcessor")
    protocol: str = PROTOCOL


class RetrieverDecodeOnlyWorkflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("wde.tasks.retriever.arg_utils:"
                       "RetrieverDecodeOnlyEngineArgs")
    protocol: str = PROTOCOL
