from typing import Optional

from pydantic import BaseModel, Field


class ShowRequest(BaseModel):
    name: str


class EmbeddingsRequest(BaseModel):
    model: str
    prompt: str
    options: Optional[dict] = Field(default_factory=dict)


class RerankerRequest(BaseModel):
    model: str
    query: str
    passage: str
    options: Optional[dict] = Field(default_factory=dict)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list = []
    options: Optional[dict] = Field(default_factory=dict)
    stream: bool = True
