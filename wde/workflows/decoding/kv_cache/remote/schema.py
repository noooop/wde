from typing import Any

from pydantic import BaseModel


class GetRequest(BaseModel):
    model: str
    block_hashs: Any


class GetResponse(BaseModel):
    block_hashs: Any
    blocks: Any


class SetRequest(BaseModel):
    model: str
    block_hashs: Any
    blocks: Any
    force: bool


class SetResponse(BaseModel):
    total: int
    exist: int


class ContainsRequest(BaseModel):
    model: str
    block_hashs: Any


class ContainsResponse(BaseModel):
    block_hashs: Any