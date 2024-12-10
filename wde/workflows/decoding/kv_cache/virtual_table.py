from typing import List, Optional

from wde.workflows.decoding.kv_cache.interfaces import Block, BlockAllocator


def chunk_list(lst, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_num_required_blocks(token_ids: List[int],
                            block_size: int,
                            num_lookahead_slots: int = 0) -> int:

    def cdiv(a: int, b: int) -> int:
        return -(a // -b)

    return cdiv(len(token_ids) + num_lookahead_slots, block_size)


class VirtualBlockTable:

    def __init__(
        self,
        block_size: int,
        block_allocator: BlockAllocator,
        _blocks: Optional[List[Block]] = None,
    ):

        if _blocks is None:
            _blocks: List[Block] = []

        self._blocks = _blocks
        self._block_size = block_size
        self._allocator = block_allocator
        self._num_token_ids = 0

    def __len__(self):
        return len(self._blocks)

    def allocate(self, token_ids: List[int]):
        assert not self._blocks and self._num_token_ids == 0

        block_token_ids = []
        tail_token_ids = []
        for cur_token_ids in chunk_list(token_ids, self._block_size):
            if len(cur_token_ids) == self._block_size:
                block_token_ids.append(cur_token_ids)
            else:
                tail_token_ids.append(cur_token_ids)

        prev_block = None

        if block_token_ids:
            for _ in block_token_ids:
                self._blocks.append(self._allocator.allocate_block(prev_block))
            prev_block = self._blocks[-1]

        if tail_token_ids:
            assert len(tail_token_ids) == 1
            self._blocks.append(self._allocator.allocate_block(prev_block))

        self._num_token_ids = len(token_ids)

    def append_token_ids(self, token_ids):
        num_required_blocks = get_num_required_blocks(token_ids,
                                                      self._block_size)
        num_new_blocks_needed = num_required_blocks - len(self._blocks)

        prev_block = None

        for _ in range(num_new_blocks_needed):
            self._blocks.append(self._allocator.allocate_block(prev_block))
            prev_block = self._blocks[-1]

        self._num_token_ids = len(token_ids)

    def free(self):
        for block in self._blocks:
            self._allocator.free(block)

    @property
    def physical_block_ids(self):
        return [block.block_id for block in self._blocks]
