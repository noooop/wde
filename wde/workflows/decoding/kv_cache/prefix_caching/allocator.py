from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, List, Optional

import numpy as np

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.logic_manager import (
    BlockAllocatorInterface, BlockId, NoFreeBlocksError,
    VirtualBlockTableInterface)
from wde.workflows.decoding.kv_cache.prefix_caching.lru_evictor import (
    LRUEvictor, Node)
from wde.workflows.decoding.kv_cache.prefix_caching.util import (
    DeltaTokenIDs, PrefixHash, TokenIDs, get_block_hash, get_prefix_hash)
from wde.workflows.decoding.kv_cache.utils import chunk_list

logger = init_logger(__name__)


@dataclass
class Block(Node):
    # Previous block prefix_hash
    prefix_hash: Optional[PrefixHash] = None

    # block_hash = get_block_hash(prefix_hash, delta_token_ids)
    # this block prefix_hash if is_full_block
    # full blocks map use block_hash
    block_hash: Optional[PrefixHash] = None

    delta_token_ids: Optional[DeltaTokenIDs] = None
    block_size: int = 0
    num_token_ids: int = 0
    num_computed_tokens: int = 0

    physical_block_id: Optional[BlockId] = None
    ref_count: int = 0
    lock: bool = False

    def set_token_ids(self, token_ids: TokenIDs):
        if self.delta_token_ids is None:
            self.delta_token_ids = np.zeros(self.block_size, dtype=np.int64)

        num_token_ids = len(token_ids)
        assert num_token_ids > self.num_token_ids

        self.delta_token_ids[:num_token_ids] = token_ids
        self.num_token_ids = num_token_ids

    def get_delta_token_ids(self):
        return self.delta_token_ids[:self.num_token_ids].tolist()

    def is_full_block(self):
        return self.num_token_ids == self.block_size

    def is_full_and_computed(self):
        return self.num_token_ids == self.num_computed_tokens == self.block_size

    @property
    def num_empty_slots(self):
        return self.block_size - self.num_token_ids

    def ensure_block_hash(self):
        if self.block_hash is None:
            self.block_hash = get_block_hash(self.prefix_hash,
                                             self.delta_token_ids)

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
        return self.ref_count

    def decr(self):
        assert self.ref_count > 0
        self.ref_count -= 1
        return self.ref_count


class PrefixCachingVirtualBlockTable(VirtualBlockTableInterface):
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
        init_prefix_hash: PrefixHash,
        _blocks: Optional[List[Block]] = None,
    ):
        if _blocks is None:
            _blocks: List[Block] = []

        self._blocks = _blocks
        self._block_size = block_size
        self._allocator = block_allocator
        self._num_token_ids = 0
        self._seq_len = 0
        self._init_prefix_hash = init_prefix_hash

        # cached
        # compute by _update_num_computed_tokens
        self._num_computed_tokens = 0
        self._head = 0
        self._tail = 0

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
        # Force recompute last token
        return max(0, min(self._num_token_ids - 1, self._num_computed_tokens))

    @property
    def physical_block_ids(self):
        _physical_block_ids = []
        for block in self._blocks:

            physical_block_id = block.physical_block_id
            if physical_block_id is None:
                break

            _physical_block_ids.append(physical_block_id)

        return _physical_block_ids

    #####################################
    # lock

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

    #####################################
    # update -> allocate -> update_num_computed_tokens

    def update(self, token_ids: TokenIDs):
        num_token_ids = len(token_ids)
        num_token_ids_curr = self.num_token_ids

        if num_token_ids != num_token_ids_curr:
            self._update(token_ids)
            self._num_token_ids = num_token_ids

        self._update_num_computed_tokens()

    def allocate(self, token_budget: int):
        num_new_tokens = self.num_token_ids - self.seq_len

        assert num_new_tokens > 0

        block = self._blocks[self._tail]

        if block.physical_block_id is None:
            self._allocator.allocate(block)
            num_empty_slots = self._block_size
        else:
            num_empty_slots = self._block_size - block.num_computed_tokens
            if num_empty_slots == 0:
                self._tail += 1
                block = self._blocks[self._tail]
                self._allocator.allocate(block)
                num_empty_slots = self._block_size

        self._tail += 1
        token_chunk_size = min(token_budget, num_empty_slots, num_new_tokens)
        self._seq_len += token_chunk_size
        return token_chunk_size

    def update_num_computed_tokens(self):
        seq_len = self.seq_len
        end = self._tail + 1 if self._head == self._tail else self._tail

        for i in range(self._head, end):
            self._blocks[i].num_computed_tokens = min(
                seq_len - i * self._block_size, self._block_size)
            self._blocks[i].release()

    #####################################
    # free

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

    #####################################
    # use for swap_in & swap_out

    def new_full_blocks(self):
        _new_full_blocks = []
        block_size = self._block_size
        seq_len = self.seq_len
        end = self._tail + 1 if self._head == self._tail else self._tail

        for i in range(self._head, end):
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

    #####################################
    # helper function

    def _update(self, token_ids: TokenIDs):
        num_token_ids = len(token_ids)
        num_blocks_curr = len(self._blocks)
        num_token_ids_curr = self.num_token_ids

        if num_blocks_curr == 0:
            # Start from scratch
            self._create_blocks(token_ids, prefix_hash=self._init_prefix_hash)
            return

        # deal with last block
        last_block = self._blocks[-1]
        num_empty_slots = last_block.num_empty_slots

        if num_empty_slots > 0:
            offset = (len(self._blocks) - 1) * self._block_size
            delta_token_ids = token_ids[offset:offset + self._block_size]

            last_block.set_token_ids(delta_token_ids)

            if len(delta_token_ids) == self._block_size:
                # become full block
                new_last_block = self._allocator.update(last_block)

                if new_last_block is not last_block:
                    # last_block has been released
                    self._allocator.hold(new_last_block)
                    self._blocks[-1] = new_last_block

        if num_empty_slots >= num_token_ids - num_token_ids_curr:
            # No need to create new blocks
            return

        prefix_hash = self._blocks[-1].block_hash

        offset = len(self._blocks) * self._block_size
        token_ids = token_ids[offset:]
        self._create_blocks(token_ids, prefix_hash=prefix_hash)

    def _create_blocks(self, token_ids: TokenIDs, prefix_hash: PrefixHash):
        block_size = self._block_size
        block_allocator = self._allocator

        for delta_token_ids in chunk_list(token_ids, block_size):
            if len(delta_token_ids) == block_size:
                # full_block
                delta_token_ids = np.array(delta_token_ids, dtype=np.int64)
                block_hash = get_block_hash(prefix_hash, delta_token_ids)
                block = block_allocator.get_full_block(prefix_hash, block_hash,
                                                       delta_token_ids)
                prefix_hash = block_hash
            else:
                # last portion block
                block = Block(prefix_hash=prefix_hash,
                              block_hash=None,
                              block_size=block_size)
                block.set_token_ids(delta_token_ids)
            self._allocator.hold(block)
            self._blocks.append(block)

    def _update_num_computed_tokens(self):
        self._num_computed_tokens = 0
        self._head = 0

        for i, block in enumerate(self._blocks):
            self._num_computed_tokens += block.num_computed_tokens

            if block.physical_block_id is None:
                break

            if block.num_computed_tokens != self._block_size:
                break

            self._head += 1

        # fix corner case. When all tokens hit the cache,
        # and the token is multiple of block_size
        self._head = min(self._head, len(self._blocks) - 1)
        self._tail = self._head

        # update hit the cache block
        self._seq_len = min(self.num_token_ids, self._num_computed_tokens)


class PrefixCachingBlockAllocator(BlockAllocatorInterface):

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 model_name: str,
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

        self._init_prefix_str = f"kv_cache:{model_name}:{self._block_size}"
        self._init_prefix_hash = get_prefix_hash(
            self._init_prefix_str.encode())

    def create(self):
        return PrefixCachingVirtualBlockTable(
            block_size=self._block_size,
            block_allocator=self,
            init_prefix_hash=self._init_prefix_hash)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids) + len(self._free_full_blocks)

    def allocate(self, block: Block):
        if block.physical_block_id is not None:
            return

        physical_block_id = self._get_free_physical_block_id()
        block.physical_block_id = physical_block_id

    def get_full_block(self, prefix_hash: PrefixHash, block_hash: PrefixHash,
                       delta_token_ids: DeltaTokenIDs):
        block_size = self._block_size
        assert len(delta_token_ids) == block_size

        block = self._full_blocks_map.get(block_hash, None)

        if block is None:
            block = Block(
                prefix_hash=prefix_hash,
                block_hash=block_hash,
                delta_token_ids=delta_token_ids,
                block_size=block_size,
                num_token_ids=block_size,
            )
            self._full_blocks_map[block_hash] = block
        return block

    def hold(self, block: Block):
        ref_count = block.incr()

        if ref_count == 1:
            self._remove_from_free_blocks(block)

    def free(self, block: Block):
        ref_count = block.decr()

        if ref_count > 0:
            return

        if not block.is_full_block():
            if block.physical_block_id is not None:
                self._free_physical_block_ids.append(block.physical_block_id)
        else:
            if block.physical_block_id is not None:
                assert self._full_blocks_map[block.block_hash] is block
                self._free_full_blocks.add(block)
            else:
                self._full_blocks_map.pop(block.block_hash, None)

    def update(self, block: Block):
        return self._maybe_update_full_block(block)

    def _maybe_update_full_block(self, block: Block):
        if not block.is_full_block():
            return block

        full_block = block

        full_block.ensure_block_hash()
        block_hash = full_block.block_hash

        block = self._full_blocks_map.get(block_hash, None)
        if block is full_block:
            return block
        elif block is None:
            self._full_blocks_map[block_hash] = full_block

            return full_block
        else:
            self._free(full_block)
            return block

    def _free(self, block: Block):
        assert block.ref_count == 1
        assert block not in self._free_full_blocks

        self._free_physical_block_ids.append(block.physical_block_id)

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            pass

        full_blocks = self._free_full_blocks.evict()
        self._full_blocks_map.pop(full_blocks.block_hash, None)
        return full_blocks.physical_block_id

    def _remove_from_free_blocks(self, block):
        if block is None:
            return

        if block.is_full_block():
            self._free_full_blocks.remove(block)

    @property
    def block_size(self):
        return self._block_size

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks


class DisablePrefixCachingBlockAllocator(BlockAllocatorInterface):

    def __init__(self,
                 num_blocks: int,
                 block_size: int,
                 model_name: str,
                 block_ids: Optional[Iterable[int]] = None,
                 *args,
                 **kwargs):
        self._block_size = block_size
        self._num_blocks = num_blocks

        if block_ids is None:
            block_ids = range(num_blocks)

        self._num_blocks = num_blocks
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

        self._init_prefix_str = f"kv_cache:{model_name}:{self._block_size}"
        self._init_prefix_hash = get_prefix_hash(
            self._init_prefix_str.encode())

    def create(self):
        return PrefixCachingVirtualBlockTable(
            block_size=self._block_size,
            block_allocator=self,
            init_prefix_hash=self._init_prefix_hash)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids)

    def allocate(self, block: Block):
        if block.physical_block_id is not None:
            return

        physical_block_id = self._get_free_physical_block_id()
        block.physical_block_id = physical_block_id

    def hold(self, block: Block):
        block.incr()

    def free(self, block: Block):
        ref_count = block.decr()

        if ref_count == 0 and block.physical_block_id is not None:
            self._free_physical_block_ids.append(block.physical_block_id)

    def get_full_block(self, prefix_hash: PrefixHash, block_hash: PrefixHash,
                       delta_token_ids: TokenIDs):
        assert len(delta_token_ids) == self._block_size

        block = Block(prefix_hash=prefix_hash,
                      block_hash=block_hash,
                      block_size=self._block_size)
        block.set_token_ids(delta_token_ids)
        return block

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            raise NoFreeBlocksError()

    @property
    def block_size(self):
        return self._block_size

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks
