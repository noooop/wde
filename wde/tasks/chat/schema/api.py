from typing import Optional

from pydantic import BaseModel, Field

PROTOCOL = "chat"
CHAT_PROTOCOL = PROTOCOL


class ChatCompletionRequest(BaseModel):
    model: str
    tools: Optional[list] = None
    messages: list = Field(default_factory=list)
    options: dict = Field(default_factory=dict)
    stream: bool = True


class ChatCompletionResponse(BaseModel):
    model: str
    finish_reason: str
    content: str

    completion_tokens: int
    prompt_tokens: int
    total_tokens: int


class ChatCompletionStreamResponse(BaseModel):
    model: str
    delta_content: str
    finish_reason: Optional[str] = None


class ChatCompletionStreamResponseDone(BaseModel):
    model: str
    finish_reason: Optional[str] = None
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
