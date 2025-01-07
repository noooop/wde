import queue
import time
from collections import deque
from queue import PriorityQueue
from typing import Deque, Dict, Iterable, List, Optional, cast

import torch

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.logic_manager import (BlockAllocator,
                                                           BlockId,
                                                           NoFreeBlocksError,
                                                           PrefixHash,
                                                           VirtualBlockTable)
from wde.workflows.decoding.kv_cache.prefix_caching.allocator import Block
from wde.workflows.decoding.kv_cache.utils import (chunk_list,
                                                   get_num_required_blocks)
from wde.workflows.decoding.kv_cache.yoco.copy_on_write import CopyOnWrite
from wde.workflows.decoding.kv_cache.yoco.trie import Trie

logger = init_logger(__name__)


class YOCOVirtualBlockTable(VirtualBlockTable):
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
        block_allocator: "YOCOPrefixCachingBlockAllocator",
        _blocks: Optional[List[Block]] = None,
    ):
        if _blocks is None:
            _blocks: List[Block] = []

        self._blocks = _blocks
        self._block_size = block_size
        self._allocator = block_allocator
        self._num_token_ids = 0
        self._seq_len = 0

    #####################################
    # Some intermediate variables, as read-only properties
    @property
    def num_token_ids(self):
        return self._num_token_ids

    @property
    def num_computed_tokens(self):
        _num_computed_tokens = 0
        for block in self._blocks:
            _num_computed_tokens += block.num_computed_tokens

            if block.num_computed_tokens < self._block_size or block.physical_block_id is None:
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
    # Reading kv cache does not require lock
    # Writing to kv cache require lock
    # release lock when update_num_computed_tokens

    def ready(self) -> bool:
        context_len = self.num_computed_tokens
        block_size = self._block_size

        for i, block in enumerate(self._blocks):
            nc = block.num_computed_tokens

            if nc == block_size:
                # full block read, not require lock
                continue

            if nc + i * block_size > context_len:
                # we just need to read this block
                # not require lock
                return True

            if not block.ready():
                # We need to write this block
                # require lock
                return False

        return True

    def acquire(self):
        seq_len = self.seq_len
        block_size = self._block_size

        for i, block in enumerate(self._blocks):

            nc = block.num_computed_tokens

            if nc == block_size:
                # full block read, not require lock
                continue

            if nc + i * block_size >= seq_len:
                # we just need to read this block
                # not require lock
                return

            assert block.physical_block_id is not None
            assert block.ready()

            # We need to write this block
            # require lock
            block.acquire()

    def update_num_computed_tokens(self):
        seq_len = self.seq_len
        block_size = self._block_size

        for i, block in enumerate(self._blocks):
            nc = block.num_computed_tokens

            if nc == block_size:
                # full block read, not require lock
                continue

            if nc + i * block_size >= seq_len:
                # we just need to read this block
                # not require lock
                return

            block.release()
            num_computed_tokens = min(seq_len - i * self._block_size,
                                      self._block_size)

            assert num_computed_tokens > nc

            block.num_computed_tokens = num_computed_tokens

    #####################################

    def update(self, token_ids: List[int]):
        num_token_ids = len(token_ids)
        block_size = self._block_size
        num_token_ids_curr = self.num_token_ids
        block_allocator = self._allocator

        assert num_token_ids >= num_token_ids_curr
        num_blocks_curr = len(self._blocks)

        if num_blocks_curr == 0:
            # Start from scratch
            self._create_blocks(token_ids, prefix_hash=None)
            return

        # deal with last block
        last_block = self._blocks[-1]

        prefix_hash = last_block.prefix_hash

        offset = (len(self._blocks) - 1) * block_size
        delta_token_ids = tuple(token_ids[offset:offset + block_size])

        if not num_token_ids_curr % block_size == 0:
            # last block is not full block
            self._num_token_ids = offset

            new_last_block, num_tokens = block_allocator.get_portion_block(
                prefix_hash, delta_token_ids)

            self._num_token_ids += num_tokens

            if new_last_block is not last_block:
                last_block.decr()
                new_last_block.incr()

            self._blocks[-1] = new_last_block

            if num_tokens < len(delta_token_ids):
                # wait at this block, until being computed
                return

        if num_token_ids <= len(self._blocks) * self._block_size:
            # No need to create new blocks
            return

        prefix_hash = self._blocks[-1].self_prefix_hash
        offset = len(self._blocks) * self._block_size
        token_ids = token_ids[offset:]
        self._create_blocks(token_ids, prefix_hash=prefix_hash)

    def _create_blocks(self, token_ids: List[int], prefix_hash: PrefixHash):
        block_size = self._block_size
        block_allocator = self._allocator

        stop = False

        for delta_token_ids in chunk_list(token_ids, block_size):
            if stop:
                break

            delta_token_ids = tuple(delta_token_ids)

            if len(delta_token_ids) == block_size:
                # full_block
                self_prefix_hash = hash((prefix_hash, delta_token_ids))
                block = block_allocator.get_full_block(self_prefix_hash,
                                                       delta_token_ids)

                if block is not None:
                    # match full_block
                    num_tokens = block_size
                else:
                    block, num_tokens = block_allocator.get_portion_block(
                        prefix_hash, delta_token_ids)

                    if num_tokens < block_size:
                        # wait at this block, until being computed
                        stop = True
                prefix_hash = self_prefix_hash
            else:
                # last portion block
                block, num_tokens = block_allocator.get_portion_block(
                    prefix_hash, delta_token_ids)

            self._num_token_ids += num_tokens
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

    #####################################

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


class YOCOPrefixCachingBlockAllocator(BlockAllocator):

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        kv_cache: List[torch.tensor],
        block_ids: Optional[Iterable[int]] = None,
    ):
        self._block_size = block_size
        self._num_blocks = num_blocks

        self._full_blocks_map: Dict[PrefixHash, Block] = {}
        self._portion_blocks_tries: Dict[PrefixHash, Trie] = {}
        self._free_blocks: PriorityQueue[Block] = PriorityQueue()

        if block_ids is None:
            block_ids = range(num_blocks)

        self._num_blocks = num_blocks
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)
        self._cow_thread = CopyOnWrite(kv_cache)

    def create_vblock(self):
        return YOCOVirtualBlockTable(block_size=self._block_size,
                                     block_allocator=self)

    @property
    def block_size(self):
        return self._block_size

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids) + self._free_blocks.qsize()

    def allocate_block(self, block: Block):
        if block.physical_block_id is not None:
            return

        physical_block_id = self._get_free_physical_block_id()
        block.physical_block_id = physical_block_id

    def get_full_block(self, self_prefix_hash, delta_token_ids):
        assert len(delta_token_ids) == self._block_size

        block = self._full_blocks_map.get(self_prefix_hash, None)
        return block

    def get_portion_block(self, prefix_hash, delta_token_ids):
        trie = self._get_or_create_portion_blocks_trie(prefix_hash)
        hit, candidates = trie.find(delta_token_ids)

        num_tokens = len(delta_token_ids)

        # 1. empty list: not candidates
        if not candidates:
            block = Block(delta_token_ids=delta_token_ids,
                          prefix_hash=prefix_hash,
                          _block_size=self._block_size)
            block = self._update_block(block, trie=trie)
            return block, num_tokens

        block = cast(Block, candidates[0])
        block_num_tokens = len(block.delta_token_ids)

        assert hit <= num_tokens
        assert hit <= block_num_tokens

        if block.num_computed_tokens < hit:
            # 2. wait at this block, until being computed
            return block, hit

        if hit == num_tokens:
            # 3. Match tokens longer than prefix: hit computed
            return block, num_tokens

        # hit < num_tokens
        if hit == block_num_tokens:
            # 4. append to this block
            block.delta_token_ids = delta_token_ids
            block = self._update_block(block, trie=trie)
            return block, num_tokens

        # hit < block_num_tokens
        # 5. need cow
        block_new = self._cow(block, prefix_hash, delta_token_ids, hit)
        block_new = self._update_block(block_new, trie=trie)
        return block_new, num_tokens

    def _cow(self, block, prefix_hash, delta_token_ids, n_token):
        assert n_token < len(block.delta_token_ids)

        num_computed_tokens = min(n_token, block.num_computed_tokens)

        try:
            physical_block_id = self._get_free_physical_block_id()
        except NoFreeBlocksError():
            num_computed_tokens = 0
            physical_block_id = None

        block_new = Block(delta_token_ids=delta_token_ids,
                          prefix_hash=prefix_hash,
                          _block_size=self._block_size,
                          physical_block_id=physical_block_id,
                          num_computed_tokens=num_computed_tokens)

        if num_computed_tokens > 0:
            self._cow_thread.submit(block, block_new, num_computed_tokens)
        return block_new

    def _get_or_create_portion_blocks_trie(self, prefix_hash):
        trie = self._portion_blocks_tries.get(prefix_hash, None)
        if trie is None:
            trie = Trie()
            self._portion_blocks_tries[prefix_hash] = trie

        return trie

    def _maybe_update_full_block(self, block: Block):
        if not block.is_full_block():
            return block

        full_block = block

        if full_block.self_prefix_hash is None:
            full_block.self_prefix_hash = hash(
                (full_block.prefix_hash, full_block.delta_token_ids))

        self_prefix_hash = full_block.self_prefix_hash

        block = self._full_blocks_map.get(self_prefix_hash, None)
        if block is full_block:
            return full_block
        elif block is None:
            self._full_blocks_map[self_prefix_hash] = full_block
            return full_block
        else:
            self.free(full_block)
            return block

    def _update_block(self, block: Block, trie=None):
        block = self._maybe_update_full_block(block)

        if trie is None:
            trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)

        trie.insert(block.delta_token_ids, block)
        return block

    def _free_block_and_get_physical_block_id(self, block: Block):
        assert block.physical_block_id is not None
        assert block.ref_count == 0

        if block.is_full_block():
            assert block.self_prefix_hash is not None
            del self._full_blocks_map[block.self_prefix_hash]

        trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)
        trie.delete(block.delta_token_ids, block)

        return block.physical_block_id

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            pass

        try:
            block = self._free_blocks.get()
            return self._free_block_and_get_physical_block_id(block)
        except queue.Empty:
            raise NoFreeBlocksError()

    def free(self, block: Block) -> None:
        ref_count = block.decr()

        if ref_count > 0:
            return

        if block.is_full_block():
            if block.self_prefix_hash in self._full_blocks_map:
                block.last_accessed_ts = time.time()
                self._free_blocks.put(block)
            else:
                # Found the same full_block
                trie = self._get_or_create_portion_blocks_trie(
                    block.prefix_hash)
                trie.delete(block.delta_token_ids, block)
                self._free_physical_block_ids.append(block.physical_block_id)
        else:
            trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)
            hit, candidates = trie.find(block.delta_token_ids)
            if len(candidates) == 1:
                block.last_accessed_ts = time.time()
                self._free_blocks.put(block)
            else:
                # There's no need to keep so many candidates
                trie.delete(block.delta_token_ids, block)
                self._free_physical_block_ids.append(block.physical_block_id)

    def join(self):
        self._cow_thread.join()
