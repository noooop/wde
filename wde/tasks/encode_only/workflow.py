from wde.tasks.prefill_only.workflow import PrefillOnlyWorkflow
from wde.tasks.retriever.schema.api import PROTOCOL


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("wde.tasks.encode_only.arg_utils:"
                       "EncodeOnlyEngineArgs")
    OutputProcessor: str = ("wde.tasks.encode_only.processor."
                            "output_processor:EncodeOnlyOutputProcessor")
    attn_type: str = "ENCODER"
    protocol: str = PROTOCOL
