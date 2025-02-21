from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, cast

import torch

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.logic_manager import (
    BlockAllocatorInterface, BlockId, NoFreeBlocksError,
    VirtualBlockTableInterface)
from wde.workflows.decoding.kv_cache.prefix_caching.allocator import Block
from wde.workflows.decoding.kv_cache.prefix_caching.lru_evictor import \
    LRUEvictor
from wde.workflows.decoding.kv_cache.prefix_caching.util import (
    PrefixHash, TokenIDs, get_block_hash, get_prefix_hash)
from wde.workflows.decoding.kv_cache.utils import (chunk_list,
                                                   get_num_required_blocks)
from wde.workflows.decoding.kv_cache.yoco.copy_on_write import CopyOnWrite
from wde.workflows.decoding.kv_cache.yoco.trie import Trie

logger = init_logger(__name__)


class YOCOVirtualBlockTable(VirtualBlockTableInterface):
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

    def get_computed_offset(self, token_chunk_size=None):
        num_computed_tokens = self.num_computed_tokens

        if token_chunk_size is not None:
            # new token ids should have been appended to blocks
            assert num_computed_tokens + token_chunk_size <= self.num_token_ids

            seq_len = num_computed_tokens + token_chunk_size
        else:
            seq_len = self.seq_len

        max_num_blocks = get_num_required_blocks(seq_len, self._block_size)
        assert max_num_blocks <= len(self._blocks)

        last_computed_block = max(
            0,
            get_num_required_blocks(num_computed_tokens, self._block_size) - 1)

        return last_computed_block, max_num_blocks, num_computed_tokens, seq_len

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

    def update(self, token_ids: TokenIDs):
        num_token_ids = len(token_ids)
        block_size = self._block_size
        num_token_ids_curr = self.num_token_ids
        block_allocator = self._allocator

        assert num_token_ids >= num_token_ids_curr
        num_blocks_curr = len(self._blocks)

        if num_blocks_curr == 0:
            # Start from scratch
            self._create_blocks(token_ids, prefix_hash=self._init_prefix_hash)
            return

        # deal with last block
        last_block = self._blocks[-1]

        prefix_hash = last_block.prefix_hash

        offset = (len(self._blocks) - 1) * block_size
        delta_token_ids = token_ids[offset:offset + block_size]

        if not num_token_ids_curr % block_size == 0:
            # last block is not full block
            self._num_token_ids = offset

            new_last_block, num_tokens = block_allocator.get_portion_block(
                prefix_hash, delta_token_ids)

            self._num_token_ids += num_tokens

            if new_last_block is not last_block:
                block_allocator.hold(new_last_block)
                block_allocator.free(last_block)

            self._blocks[-1] = new_last_block

            if num_tokens < len(delta_token_ids):
                # wait at this block, until being computed
                return

        if num_token_ids <= len(self._blocks) * self._block_size:
            # No need to create new blocks
            return

        prefix_hash = self._blocks[-1].block_hash
        offset = len(self._blocks) * self._block_size
        token_ids = token_ids[offset:]
        self._create_blocks(token_ids, prefix_hash=prefix_hash)

    def _create_blocks(self, token_ids: TokenIDs, prefix_hash: PrefixHash):
        block_size = self._block_size
        block_allocator = self._allocator

        stop = False

        for delta_token_ids in chunk_list(token_ids, block_size):
            if stop:
                break

            if len(delta_token_ids) == block_size:
                # full_block
                block_hash = get_block_hash(prefix_hash, delta_token_ids)
                block = block_allocator.get_full_block(block_hash,
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
                prefix_hash = block_hash
            else:
                # last portion block
                block, num_tokens = block_allocator.get_portion_block(
                    prefix_hash, delta_token_ids)

            self._num_token_ids += num_tokens

            block_allocator.hold(block)
            self._blocks.append(block)

    def can_allocate(self, token_chunk_size: int):
        (last_computed_block, max_num_blocks, num_computed_tokens,
         seq_len) = self.get_computed_offset(token_chunk_size)

        last_block = self._blocks[last_computed_block]

        num_empty_slots = last_block.num_empty_slots

        if num_empty_slots >= token_chunk_size:
            # No allocate required
            return token_chunk_size

        num_need_allocated_blocks = 0
        for i in range(last_computed_block, max_num_blocks):
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
        (last_computed_block, max_num_blocks, num_computed_tokens,
         seq_len) = self.get_computed_offset(token_chunk_size)

        for i in range(last_computed_block, max_num_blocks):
            self._allocator.allocate(self._blocks[i])

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

    def new_full_blocks(self):
        _new_full_blocks = []
        block_size = self._block_size
        (last_computed_block, max_num_blocks, num_computed_tokens,
         seq_len) = self.get_computed_offset()

        for i in range(last_computed_block, max_num_blocks):
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


class YOCOPrefixCachingBlockAllocator(BlockAllocatorInterface):

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        model_name: str,
        kv_cache: List[torch.tensor],
        block_ids: Optional[Iterable[int]] = None,
    ):
        self._block_size = block_size
        self._num_blocks = num_blocks

        if block_ids is None:
            block_ids = range(num_blocks)

        self._full_blocks_map: Dict[PrefixHash, Block] = {}
        self._portion_blocks_tries: Dict[PrefixHash, Trie] = {}

        self._free_blocks = LRUEvictor()
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

        self._init_prefix_str = f"kv_cache:{model_name}:{self._block_size}"
        self._init_prefix_hash = get_prefix_hash(
            self._init_prefix_str.encode())

        self._cow_thread = CopyOnWrite(kv_cache=kv_cache, block_allocator=self)

    def create(self):
        return YOCOVirtualBlockTable(block_size=self._block_size,
                                     block_allocator=self,
                                     init_prefix_hash=self._init_prefix_hash)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids) + len(self._free_blocks)

    def allocate(self, block: Block):
        if block.physical_block_id is not None:
            return

        physical_block_id = self._get_free_physical_block_id()
        block.physical_block_id = physical_block_id

    def hold(self, block: Block):
        ref_count = block.incr()

        if ref_count == 1:
            self._remove_from_free_blocks(block)

    def free(self, block: Block):
        ref_count = block.decr()

        if ref_count > 0:
            return

        if block.is_full_block():
            if block.physical_block_id is not None:
                assert self._full_blocks_map[block.block_hash] is block
                self._free_blocks.add(block)
            else:
                self._full_blocks_map.pop(block.block_hash, None)
                trie = self._get_or_create_portion_blocks_trie(
                    block.prefix_hash)
                trie.delete(block.delta_token_ids, block)
        else:
            trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)
            hit, candidates = trie.find(block.delta_token_ids)
            if len(candidates) == 1:
                if block.physical_block_id is not None:
                    self._free_blocks.add(block)
            else:
                # There's no need to keep so many candidates
                trie.delete(block.delta_token_ids, block)

                if block.physical_block_id is not None:
                    self._free_physical_block_ids.append(
                        block.physical_block_id)

    def update(self, block: Block, trie=None):
        block = self._maybe_update_full_block(block, trie=trie)

        if trie is None:
            trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)

        trie.insert(block.delta_token_ids, block)
        return block

    def join(self):
        self._cow_thread.join()

    def get_full_block(self, block_hash: PrefixHash,
                       delta_token_ids: TokenIDs):
        assert len(delta_token_ids) == self._block_size

        block = self._full_blocks_map.get(block_hash, None)
        return block

    def get_portion_block(self, prefix_hash: PrefixHash,
                          delta_token_ids: TokenIDs):
        trie = self._get_or_create_portion_blocks_trie(prefix_hash)
        hit, candidates = trie.find(delta_token_ids)

        num_tokens = len(delta_token_ids)

        # 1. empty list: not candidates
        if not candidates:
            block = Block(delta_token_ids=delta_token_ids,
                          prefix_hash=prefix_hash,
                          _block_size=self._block_size)
            block = self.update(block, trie=trie)
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
            block = self.update(block, trie=trie)
            return block, num_tokens

        # hit < block_num_tokens
        # 5. need cow
        block_new = self._cow(block, prefix_hash, delta_token_ids, hit)
        block_new = self.update(block_new, trie=trie)
        return block_new, num_tokens

    @property
    def block_size(self):
        return self._block_size

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    def _cow(self, block: Block, prefix_hash: PrefixHash,
             delta_token_ids: TokenIDs, n_token: int):
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

    def _maybe_update_full_block(self, block: Block, trie: Trie = None):
        if not block.is_full_block():
            return block

        full_block = block

        full_block.ensure_block_hash()

        block_hash = full_block.block_hash

        block = self._full_blocks_map.get(block_hash, None)
        if block is full_block:
            return full_block
        elif block is None:
            self._full_blocks_map[block_hash] = full_block
            return full_block
        else:
            self._free(full_block)
            return block

    def _free(self, block: Block, trie: Trie = None):
        assert block.ref_count == 1
        assert block not in self._free_blocks

        if trie is None:
            trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)

        trie.delete(block.delta_token_ids, block)

        if block.physical_block_id is not None:
            self._free_physical_block_ids.append(block.physical_block_id)

    def _free_block_and_get_physical_block_id(self, block: Block):
        assert block.ref_count == 0

        if block.is_full_block():
            assert block.block_hash is not None
            del self._full_blocks_map[block.block_hash]

        trie = self._get_or_create_portion_blocks_trie(block.prefix_hash)
        trie.delete(block.delta_token_ids, block)

        return block.physical_block_id

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            pass

        block = self._free_blocks.evict()
        return self._free_block_and_get_physical_block_id(block)

    def _remove_from_free_blocks(self, block: Block):
        if block is None:
            return

        self._free_blocks.remove(block)
