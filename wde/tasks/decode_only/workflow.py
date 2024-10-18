from typing import Dict

from wde.tasks.core.workflow import Workflow
from wde.tasks.decode_only.output_last_hidden_states.workflow import \
    DecodeOnlyOutputLastHiddenStatesWorkflow


class DecodeOnlyWorkflow(Workflow):

    @classmethod
    def from_engine_args(cls, engine_args: Dict):
        if engine_args.get("output_last_hidden_states", False):
            return DecodeOnlyOutputLastHiddenStatesWorkflow
        raise NotImplementedError(
            "Generating models is not supported temporarily")
