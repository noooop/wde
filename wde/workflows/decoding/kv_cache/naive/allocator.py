from collections import deque
from typing import Deque, Iterable, List, Optional

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.logic_manager import (
    BlockAllocatorInterface, BlockId, NoFreeBlocksError,
    VirtualBlockTableInterface)

logger = init_logger(__name__)


class NaiveVirtualBlockTable(VirtualBlockTableInterface):

    def __init__(
        self,
        block_size: int,
        block_allocator: BlockAllocatorInterface,
        _physical_block_ids: Optional[List[BlockId]] = None,
    ):
        if _physical_block_ids is None:
            _physical_block_ids: List[BlockId] = []

        self._physical_block_ids = _physical_block_ids
        self._block_size = block_size
        self._allocator = block_allocator
        self._num_token_ids = 0
        self._seq_len = 0
        self._num_computed_tokens = 0

    #####################################
    # Some intermediate variables, as read-only properties

    @property
    def num_token_ids(self):
        return self._num_token_ids

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def num_computed_tokens(self):
        return max(0, min(self._num_token_ids - 1, self._num_computed_tokens))

    @property
    def physical_block_ids(self):
        return self._physical_block_ids

    #####################################
    # lock

    def ready(self) -> bool:
        # naive BlockTable will not be locked
        return True

    def acquire(self):
        return

    def release(self):
        return

    #####################################
    # update -> allocate -> update_num_computed_tokens

    def update(self, token_ids: List[int]):
        num_token_ids = len(token_ids)
        num_token_ids_curr = self.num_token_ids

        assert num_token_ids >= num_token_ids_curr

        self._num_token_ids = num_token_ids

    def allocate(self, token_budget: int):
        num_new_tokens = self.num_token_ids - self.seq_len
        num_empty_slots = self.num_empty_slots

        assert num_new_tokens > 0

        if num_empty_slots == 0:
            # no empty_slot
            # try allocate a new block
            physical_block_id = self._allocator.allocate()
            self._physical_block_ids.append(physical_block_id)

            num_empty_slots = self._block_size

        token_chunk_size = min(token_budget, num_empty_slots, num_new_tokens)
        self._seq_len += token_chunk_size
        return token_chunk_size

    def update_num_computed_tokens(self):
        self._num_computed_tokens = self._seq_len

    #####################################
    # free

    def free(self):
        assert self._seq_len == self._num_computed_tokens

        for physical_block_id in self._physical_block_ids:
            self._allocator.free(physical_block_id)

        self._physical_block_ids = []
        self._seq_len = 0
        self._num_computed_tokens = 0

    def free_last_block(self):
        if not self._physical_block_ids:
            return

        assert self._seq_len == self._num_computed_tokens

        last_physical_block_id = self._physical_block_ids.pop(-1)

        self._allocator.free(last_physical_block_id)
        self._seq_len = len(self._physical_block_ids) * self._block_size
        self._num_computed_tokens = min(self._seq_len,
                                        self._num_computed_tokens)

    #####################################
    # helper function

    @property
    def num_empty_slots(self):
        return len(self._physical_block_ids) * self._block_size - self._seq_len

    @property
    def max_num_token_ids(self):
        return len(self._physical_block_ids) * self._block_size


class NaiveBlockAllocator(BlockAllocatorInterface):

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 block_ids: Optional[Iterable[int]] = None,
                 *args,
                 **kwargs):
        self._block_size = block_size

        if block_ids is None:
            block_ids = range(num_blocks)

        self._num_blocks = num_blocks
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

    def create(self):
        return NaiveVirtualBlockTable(block_size=self._block_size,
                                      block_allocator=self)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids)

    def allocate(self):
        return self._get_free_physical_block_id()

    def hold(self, *args, **kwargs):
        pass

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

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks
