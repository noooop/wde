from typing import Any, Optional

from pydantic import BaseModel


class GetRequest(BaseModel):
    model: str
    stream: bool
    block_hashs: Any


class GetResponse(BaseModel):
    total: int
    hit: int
    miss: int
    block_hashs: Optional[Any] = None
    blocks: Optional[Any] = None


class GetResponseStream(BaseModel):
    block_hash: Any
    block: Any


class SetRequest(BaseModel):
    model: str
    block_hashs: Any
    blocks: Any
    force: bool
    deferred: bool


class SetResponse(BaseModel):
    total: int
    exist: int


class ContainsRequest(BaseModel):
    model: str
    refresh: bool
    block_hashs: Any


class ContainsResponse(BaseModel):
    hit: Any
    miss: Any


class InfoResponse(BaseModel):
    block_size: int
    num_blocks: int
    num_full_blocks: int
    num_free_full_blocks: int
    num_free_physical_block_ids: int