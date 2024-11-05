from dataclasses import dataclass
from typing import List

from wde.tasks.core.schema.engine_io import (PromptInput, Request,
                                             SchedulableRequest,
                                             SchedulerOutput, TextOnlyInputs)


@dataclass
class PrefillOnlyInput(TextOnlyInputs):
    pass


@dataclass
class PrefillOnlyRequest(Request):
    inputs: PromptInput


@dataclass
class PrefillOnlySchedulableRequest(SchedulableRequest):
    inputs: TextOnlyInputs = None

    @property
    def num_new_tokens(self):
        return len(self.inputs.prompt_token_ids)


@dataclass
class PrefillOnlySchedulerOutput(SchedulerOutput):
    scheduled_requests: List[SchedulableRequest]
    ignored_requests: List[SchedulableRequest]

    def is_empty(self) -> bool:
        return not self.scheduled_requests
