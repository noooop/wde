from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

from wde.tasks.core.schema.engine_io import Inputs, Request, RequestOutput


@dataclass
class Pairs(Inputs):
    query: str
    passage: str


RerankerInputs = Union[Sequence, Pairs]


@dataclass
class RerankerRequest(Request):
    inputs: Pairs


@dataclass
class RerankerRequestOutput(RequestOutput):
    prompt_token_ids: Optional[List[int]] = None
    score: Optional[float] = None
