from wde.tasks.retriever.schema.api import PROTOCOL
from wde.workflows.prefill_only.workflow import PrefillOnlyWorkflow


class EncodeOnlyWorkflow(PrefillOnlyWorkflow):
    EngineArgs: str = ("wde.tasks.encode_only.arg_utils:"
                       "EncodeOnlyEngineArgs")
    OutputProcessor: str = ("wde.tasks.encode_only.processor."
                            "output_processor:EncodeOnlyOutputProcessor")
    attn_type: str = "ENCODER"
    protocol: str = PROTOCOL
