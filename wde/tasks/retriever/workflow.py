from wde.tasks.decode_only.workflow import DecodeOnlyWorkflow
from wde.tasks.encode_only.workflow import EncodeOnlyWorkflow


class RetrieverEncodeOnlyWorkflow(EncodeOnlyWorkflow):
    OutputProcessor: str = ("wde.tasks.retriever.processor."
                            "output_processor:RetrieverOutputProcessor")


class RetrieverDecodeOnlyWorkflow(DecodeOnlyWorkflow):
    EngineArgs: str = ("wde.tasks.retriever.arg_utils:"
                       "RetrieverDecodeOnlyEngineArgs")
