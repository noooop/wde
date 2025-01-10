from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.logic_manager import (BlockAllocator,
                                                           BlockId,
                                                           NoFreeBlocksError,
                                                           PrefixHash,
                                                           VirtualBlockTable)
from wde.workflows.decoding.kv_cache.prefix_caching.lru_evictor import \
    LRUEvictor
from wde.workflows.decoding.kv_cache.utils import (chunk_list,
                                                   get_num_required_blocks)

logger = init_logger(__name__)


@dataclass
class Block:
    delta_token_ids: Optional[List[int]] = None

    # Previous block prefix_hash
    prefix_hash: Optional[PrefixHash] = None

    # self_prefix_hash = hash((prefix_hash, delta_token_ids))
    # this block prefix_hash if is_full_block
    # full blocks map use self_prefix_hash
    self_prefix_hash: Optional[PrefixHash] = None

    physical_block_id: Optional[BlockId] = None
    num_computed_tokens: int = 0

    ref_count: int = 0
    lock: bool = False
    _block_size: int = 0

    def is_full_and_computed(self):
        return self.num_computed_tokens == self._block_size

    def ready(self):
        if self.is_full_and_computed():
            return True

        return not self.lock

    def acquire(self):
        if self.ready():
            self.lock = True
            return True

        return False

    def release(self):
        if self.is_full_and_computed():
            return

        assert self.lock
        self.lock = False

    def incr(self):
        self.ref_count += 1

    def decr(self):
        assert self.ref_count > 0
        self.ref_count -= 1
        return self.ref_count

    def is_full_block(self):
        return len(self.delta_token_ids) == self._block_size

    @property
    def num_empty_slots(self):
        return self._block_size - len(self.delta_token_ids)

    def ensure_self_prefix_hash(self):
        if self.self_prefix_hash is None:
            self.self_prefix_hash = hash(
                (self.prefix_hash, self.delta_token_ids))


class PrefixCachingVirtualBlockTable(VirtualBlockTable):
    # | <-                           max capacity                                      -> |
    # | Full blocks...........                            |       last portion block      |
    # | <-           num_token_ids                            ->  | <- num_empty_slots -> |
    # |     computed        |          num_new_tokens             |
    #
    #   after chunked prefill:
    #
    # |     computed        | prepare to compute | not computed   | <- num_empty_slots -> |
    # | num_computed_tokens | token_chunk_size   |
    # | context_len         | query_len          |
    # | seq_len                                  |

    def __init__(
        self,
        block_size: int,
        block_allocator: "PrefixCachingBlockAllocator",
        _blocks: Optional[List[Block]] = None,
    ):
        if _blocks is None:
            _blocks: List[Block] = []

        self._blocks = _blocks
        self._block_size = block_size
        self._allocator = block_allocator
        self._num_token_ids = 0
        self._seq_len = 0

    @property
    def num_token_ids(self):
        return self._num_token_ids

    @property
    def num_computed_tokens(self):
        _num_computed_tokens = 0
        for block in self._blocks:
            _num_computed_tokens += block.num_computed_tokens

            if block.num_computed_tokens != self._block_size:
                break

        # Force recompute last token
        return min(self._num_token_ids - 1, _num_computed_tokens)

    @property
    def num_new_tokens(self):
        num_computed_tokens = self.num_computed_tokens
        num_token_ids = self.num_token_ids

        if num_computed_tokens == num_token_ids:
            return 1
        else:
            return num_token_ids - num_computed_tokens

    @property
    def context_len(self):
        return self.num_computed_tokens

    @property
    def seq_len(self):
        return self._seq_len

    @property
    def query_len(self):
        return self.seq_len - self.context_len

    def update(self, token_ids: List[int]):
        num_token_ids = len(token_ids)
        num_token_ids_curr = self.num_token_ids

        assert num_token_ids >= num_token_ids_curr
        num_blocks_curr = len(self._blocks)

        if num_token_ids == num_blocks_curr:
            return

        if num_blocks_curr == 0:
            # Start from scratch
            self._create_blocks(token_ids, prefix_hash=None)
            self._num_token_ids = len(token_ids)
            return

        # deal with last block
        last_block = self._blocks[-1]
        num_empty_slots = last_block.num_empty_slots

        if num_empty_slots > 0:
            offset = (len(self._blocks) - 1) * self._block_size
            delta_token_ids = tuple(token_ids[offset:offset +
                                              self._block_size])
            last_block.delta_token_ids = delta_token_ids

            if len(delta_token_ids) == self._block_size:
                # become full block
                new_last_block = self._allocator.update_block(last_block)

                if new_last_block is not last_block:
                    # last_block has been released
                    new_last_block.incr()
                    self._blocks[-1] = new_last_block

        if num_empty_slots >= num_token_ids - num_token_ids_curr:
            # No need to create new blocks
            self._num_token_ids = num_token_ids
            return

        prefix_hash = self._blocks[-1].self_prefix_hash

        offset = len(self._blocks) * self._block_size
        token_ids = token_ids[offset:]
        self._create_blocks(token_ids, prefix_hash=prefix_hash)
        self._num_token_ids = num_token_ids

    def _create_blocks(self, token_ids: List[int], prefix_hash: PrefixHash):
        block_size = self._block_size
        block_allocator = self._allocator

        for delta_token_ids in chunk_list(token_ids, block_size):
            if len(delta_token_ids) == block_size:
                # full_block
                delta_token_ids = tuple(delta_token_ids)
                self_prefix_hash = hash((prefix_hash, delta_token_ids))

                block = block_allocator.get_full_block(prefix_hash,
                                                       self_prefix_hash,
                                                       delta_token_ids)
                prefix_hash = self_prefix_hash
            else:
                # last portion block
                block = Block(delta_token_ids=delta_token_ids,
                              prefix_hash=prefix_hash,
                              self_prefix_hash=None,
                              _block_size=block_size)
            block.incr()
            self._blocks.append(block)

    def can_allocate(self, token_chunk_size: int):
        num_computed_tokens = self.num_computed_tokens
        max_num_block = get_num_required_blocks(
            num_computed_tokens + token_chunk_size, self._block_size)

        # new token ids should have been appended to blocks
        assert num_computed_tokens + token_chunk_size <= self.num_token_ids
        assert max_num_block <= len(self._blocks)

        # last portion block
        last_block_idx = max(
            0,
            get_num_required_blocks(num_computed_tokens, self._block_size) - 1)
        last_block = self._blocks[last_block_idx]

        num_empty_slots = last_block.num_empty_slots

        if num_empty_slots >= token_chunk_size:
            # No allocate required
            return token_chunk_size

        num_need_allocated_blocks = 0
        for i in range(last_block_idx, max_num_block):
            if self._blocks[i].physical_block_id is None:
                num_need_allocated_blocks += 1

        num_free_gpu_blocks = self._allocator.num_free_blocks

        if num_free_gpu_blocks > num_need_allocated_blocks:
            return token_chunk_size
        else:
            return min(
                token_chunk_size,
                num_empty_slots + num_free_gpu_blocks * self._block_size)

    def allocate(self, token_chunk_size: int):
        num_computed_tokens = self.num_computed_tokens
        seq_len = num_computed_tokens + token_chunk_size

        max_num_block = get_num_required_blocks(seq_len, self._block_size)

        # new token ids should have been appended to blocks
        assert seq_len <= self.num_token_ids
        assert max_num_block <= len(self._blocks)

        last_block_idx = get_num_required_blocks(num_computed_tokens,
                                                 self._block_size)

        for i in range(last_block_idx, max_num_block):
            self._allocator.allocate_block(self._blocks[i])

        self._seq_len = seq_len

    @property
    def physical_block_ids(self):
        _physical_block_ids = []
        for block in self._blocks:

            physical_block_id = block.physical_block_id
            if physical_block_id is None:
                break

            _physical_block_ids.append(physical_block_id)

        return _physical_block_ids

    def ready(self) -> bool:
        for block in self._blocks:
            if block.physical_block_id is None:
                # All allocated blocks are ready
                return True

            if not block.ready():
                return False

        return True

    def acquire(self):
        for block in self._blocks:
            if block.physical_block_id is None:
                return

            assert block.ready()
            block.acquire()

    def release(self):
        for block in self._blocks:
            if block.physical_block_id is None:
                return

            block.release()

    def free(self):
        for block in self._blocks:
            self._allocator.free(block)

        self._num_token_ids = 0
        self._seq_len = 0

    def free_last_block(self):
        last_block = self._blocks.pop(-1)

        self._allocator.free(last_block)
        self._num_token_ids = len(self._blocks) * self._block_size
        self._seq_len = min(self._num_token_ids, self._seq_len)

    def update_num_computed_tokens(self):
        seq_len = self.seq_len
        max_num_block = get_num_required_blocks(seq_len, self._block_size)

        # new token ids should have been appended to blocks
        assert seq_len <= self.seq_len
        assert max_num_block <= len(self._blocks)

        num_computed_tokens = self.num_computed_tokens
        last_block_idx = max(
            0,
            get_num_required_blocks(num_computed_tokens, self._block_size) - 1)

        for i in range(last_block_idx, max_num_block):
            self._blocks[i].num_computed_tokens = min(
                seq_len - i * self._block_size, self._block_size)
            self._blocks[i].release()

    def new_full_blocks(self):
        _new_full_blocks = []
        block_size = self._block_size
        seq_len = self.seq_len
        max_num_block = get_num_required_blocks(seq_len, self._block_size)

        num_computed_tokens = self.num_computed_tokens
        last_block_idx = max(
            0,
            get_num_required_blocks(num_computed_tokens, self._block_size) - 1)

        for i in range(last_block_idx, max_num_block):
            new_num_computed_tokens = min(seq_len - i * self._block_size,
                                          self._block_size)

            if self._blocks[
                    i].num_computed_tokens < block_size and new_num_computed_tokens == block_size:
                _new_full_blocks.append(self._blocks[i])

        return _new_full_blocks

    def get_maybe_swap_in_blocks(self):
        blocks = []

        for block in self._blocks:
            if not block.is_full_block():
                # 1. offloading kv cache only take cares of full blocks
                continue

            if block.is_full_and_computed():
                # 2. computed, no need to swap in
                continue

            if block.lock:
                # 3. busy block, maybe doing swap_in, maybe doing computing
                continue

            # swap_in this block if possible
            blocks.append(block)
        return blocks


class PrefixCachingBlockAllocator(BlockAllocator):

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 block_ids: Optional[Iterable[int]] = None,
                 *args,
                 **kwargs):
        self._block_size = block_size
        self._num_blocks = num_blocks

        if block_ids is None:
            block_ids = range(num_blocks)

        self._num_blocks = num_blocks
        self._full_blocks_map: Dict[PrefixHash, Block] = {}

        self._free_full_blocks = LRUEvictor()
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

    def create_vblock(self):
        return PrefixCachingVirtualBlockTable(block_size=self._block_size,
                                              block_allocator=self)

    @property
    def block_size(self):
        return self._block_size

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids) + len(self._free_full_blocks)

    def get_full_block(self, prefix_hash, self_prefix_hash, delta_token_ids):
        assert len(delta_token_ids) == self._block_size

        block = self._full_blocks_map.get(self_prefix_hash, None)

        if block is None:
            block = Block(delta_token_ids=delta_token_ids,
                          prefix_hash=prefix_hash,
                          self_prefix_hash=self_prefix_hash,
                          _block_size=self._block_size)
            self._full_blocks_map[self_prefix_hash] = block
        else:
            self._free_full_blocks.remove(block)
        return block

    def _maybe_update_full_block(self, block: Block):
        if not block.is_full_block():
            return block

        full_block = block

        full_block.ensure_self_prefix_hash()
        self_prefix_hash = full_block.self_prefix_hash

        block = self._full_blocks_map.get(self_prefix_hash, None)
        if block is full_block:
            return block
        elif block is None:
            self._full_blocks_map[self_prefix_hash] = full_block
            return full_block
        else:
            self._free(full_block)
            return block

    def update_block(self, block: Block):
        return self._maybe_update_full_block(block)

    def _free(self, block: Block):
        assert block.ref_count == 1
        assert block not in self._free_full_blocks

        self._free_physical_block_ids.append(block.physical_block_id)

    def free(self, block: Block) -> None:
        ref_count = block.decr()

        if ref_count == 0:
            if not block.is_full_block():
                self._free_physical_block_ids.append(block.physical_block_id)
            else:
                assert self._full_blocks_map[block.self_prefix_hash] is block
                self._free_full_blocks.add(block)

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            pass

        full_blocks = self._free_full_blocks.evict()
        self._full_blocks_map.pop(full_blocks.self_prefix_hash, None)
        return full_blocks.physical_block_id

    def allocate_block(self, block: Block):
        if block.physical_block_id is not None:
            return

        physical_block_id = self._get_free_physical_block_id()
        block.physical_block_id = physical_block_id


class DisablePrefixCachingBlockAllocator(BlockAllocator):

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 block_ids: Optional[Iterable[int]] = None,
                 *args,
                 **kwargs):
        self._block_size = block_size
        self._num_blocks = num_blocks

        if block_ids is None:
            block_ids = range(num_blocks)

        self._num_blocks = num_blocks
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

    def create_vblock(self):
        return PrefixCachingVirtualBlockTable(block_size=self._block_size,
                                              block_allocator=self)

    @property
    def block_size(self):
        return self._block_size

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids)

    def get_full_block(self, prefix_hash, self_prefix_hash, delta_token_ids):
        assert len(delta_token_ids) == self._block_size

        block = Block(delta_token_ids=delta_token_ids,
                      prefix_hash=prefix_hash,
                      self_prefix_hash=self_prefix_hash,
                      _block_size=self._block_size)

        return block

    def update_block(self, block: Block):
        return block

    def free(self, block: Block) -> None:
        ref_count = block.decr()

        if ref_count == 0:
            self._free_physical_block_ids.append(block.physical_block_id)

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            raise NoFreeBlocksError()

    def allocate_block(self, block: Block):
        if block.physical_block_id is not None:
            return

        physical_block_id = self._get_free_physical_block_id()
        block.physical_block_id = physical_block_id
