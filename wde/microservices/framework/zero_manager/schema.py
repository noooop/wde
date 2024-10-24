from pydantic import BaseModel, Field


class StartRequest(BaseModel):
    name: str
    engine_kwargs: dict = Field(default_factory=dict)


class TerminateRequest(BaseModel):
    name: str


class StatusRequest(BaseModel):
    name: str
