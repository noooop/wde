from wde.tasks.core.workflow import Workflow
from wde.tasks.prefill_only.workflow import PrefillOnlyWorkflow


class DecodeOnlyOutputLastHiddenStatesWorkflow(Workflow):
    EngineArgs: str = ("wde.tasks.decode_only.output_last_hidden_states."
                       "arg_utils:DecodeOnlyOutputLastHiddenStatesEngineArgs")
    attn_type: str = "DECODER"

    @classmethod
    def from_engine(cls, engine):
        workflow = PrefillOnlyWorkflow.from_engine(engine)

        if engine.engine_config.model_config.enable_bidirectional:
            workflow.attn_type = "ENCODER"
        else:
            workflow.attn_type = "DECODER"

        workflow.OutputProcessor = (
            "wde.tasks.decode_only.output_last_hidden_states."
            "processor.output_processor:"
            "DecodeOnlyHiddenStatesOutputProcessor")
        return workflow
