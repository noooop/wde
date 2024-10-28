from typing import Optional

from pydantic import BaseModel, Field

PROTOCOL = "reranker"
RERANKER_PROTOCOL = PROTOCOL


class RerankerRequest(BaseModel):
    model: str
    pairs: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


class RerankerResponse(BaseModel):
    model: str
    score: float
    metrics: Optional[dict] = None
