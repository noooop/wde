from typing import Any

from pydantic import BaseModel, Field

PROTOCOL = "retriever"
RETRIEVER_PROTOCOL = PROTOCOL


class RetrieverRequest(BaseModel):
    model: str
    inputs: str
    options: dict = Field(default_factory=dict)


class RetrieverResponse(BaseModel):
    model: str
    embedding: Any
