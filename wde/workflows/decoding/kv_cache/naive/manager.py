from collections import deque
from typing import Deque, Iterable, List, Optional

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.interfaces import (BlockAllocator,
                                                        BlockId,
                                                        NoFreeBlocksError,
                                                        VirtualBlockTable)
from wde.workflows.decoding.kv_cache.utils import get_num_required_blocks
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

logger = init_logger(__name__)


class NaiveKVCacheManager:

    def __init__(self, engine_config):
        self.engine_config = engine_config
        num_gpu_blocks = self.engine_config.cache_config.num_gpu_blocks
        self._block_size = self.engine_config.cache_config.block_size
        self.block_allocator = NaiveBlockAllocator(num_blocks=num_gpu_blocks,
                                                   block_size=self._block_size)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine_config=engine.engine_config)

    def create_vblock(self, request: DecodingSchedulableRequest):
        request.vblock = self.block_allocator.create_vblock()

    def can_allocate(self, request: DecodingSchedulableRequest,
                     budget_bound_token_chunk_size: int) -> int:
        num_empty_slots = request.vblock.num_empty_slots

        # No allocat required
        if num_empty_slots >= budget_bound_token_chunk_size:
            return budget_bound_token_chunk_size

        num_need_allocated_slots = budget_bound_token_chunk_size - num_empty_slots
        num_need_allocated_blocks = get_num_required_blocks(
            num_need_allocated_slots, self._block_size)

        num_free_gpu_blocks = self.block_allocator.num_free_blocks

        if num_free_gpu_blocks > num_need_allocated_blocks:
            return budget_bound_token_chunk_size
        else:
            return min(
                budget_bound_token_chunk_size,
                num_empty_slots + num_free_gpu_blocks * self._block_size)

    def allocate(self, request: DecodingSchedulableRequest) -> None:
        token_ids = request.get_token_ids()
        offset = request.vblock.num_token_ids + request.token_chunk_size
        request.vblock.allocate(token_ids[:offset])
        assert request.vblock.num_token_ids == request.vblock.num_computed_tokens + request.token_chunk_size

    def free(self, request: DecodingSchedulableRequest) -> None:
        request.vblock.free()

    def free_last_block(self, request: DecodingSchedulableRequest):
        request.vblock.free_last_block()
        request.num_preempted += 1


class NaiveVirtualBlockTable(VirtualBlockTable):

    def __init__(
        self,
        block_size: int,
        block_allocator: BlockAllocator,
        _physical_block_ids: Optional[List[BlockId]] = None,
    ):
        if _physical_block_ids is None:
            _physical_block_ids: List[BlockId] = []

        self._physical_block_ids = _physical_block_ids
        self._block_size = block_size
        self._allocator = block_allocator
        self._num_token_ids = 0
        self._num_computed_tokens = 0

    @property
    def physical_block_ids(self):
        return self._physical_block_ids

    @property
    def num_token_ids(self):
        return self._num_token_ids

    @property
    def num_computed_tokens(self):
        return self._num_computed_tokens

    def __len__(self):
        return len(self._physical_block_ids)

    @property
    def num_empty_slots(self):
        return len(
            self._physical_block_ids) * self._block_size - self._num_token_ids

    @property
    def max_num_token_ids(self):
        return len(self._physical_block_ids) * self._block_size

    def allocate(self, token_ids):
        num_token_ids = len(token_ids)
        max_num_token_ids = self.max_num_token_ids

        # No allocate required
        if num_token_ids <= max_num_token_ids:
            self._num_token_ids = num_token_ids
            return

        num_need_allocated_slots = num_token_ids - max_num_token_ids
        num_need_allocated_blocks = get_num_required_blocks(
            num_need_allocated_slots, self._block_size)

        for i in range(num_need_allocated_blocks):
            physical_block_id = self._allocator.allocate_block()
            self._physical_block_ids.append(physical_block_id)

        self._num_token_ids = num_token_ids

    def free(self):
        assert self._num_token_ids == self._num_computed_tokens

        for physical_block_id in self._physical_block_ids:
            self._allocator.free(physical_block_id)
        self._num_token_ids = 0
        self._num_computed_tokens = 0

    def free_last_block(self):
        assert self._num_token_ids == self._num_computed_tokens
        if not self._physical_block_ids:
            return

        last_physical_block_id = self._physical_block_ids.pop(-1)
        self._allocator.free(last_physical_block_id)
        self._num_token_ids = len(self._physical_block_ids) * self._block_size
        self._num_computed_tokens = self._num_token_ids

    def update_num_computed_tokens(self):
        self._num_computed_tokens = self._num_token_ids


class NaiveBlockAllocator(BlockAllocator):

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 block_ids: Optional[Iterable[int]] = None):
        self._block_size = block_size

        if block_ids is None:
            block_ids = range(num_blocks)

        self._num_blocks = num_blocks
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids)

    def create_vblock(self):
        return NaiveVirtualBlockTable(block_size=self._block_size,
                                      block_allocator=self)

    def allocate_block(self):
        return self._get_free_physical_block_id()

    def free(self, physical_block_id: BlockId) -> None:
        self._put_physical_block_id(physical_block_id)

    def _get_free_physical_block_id(self):
        if not self._free_physical_block_ids:
            raise NoFreeBlocksError()

        physical_block_id = self._free_physical_block_ids.popleft()
        return physical_block_id

    def _put_physical_block_id(self, physical_block_id: BlockId) -> None:
        assert physical_block_id < self._num_blocks

        self._free_physical_block_ids.appendleft(physical_block_id)
