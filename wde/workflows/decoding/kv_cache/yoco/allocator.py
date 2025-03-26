from typing import Dict, Iterable, List, Optional, cast

import torch

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.logic_manager import NoFreeBlocksError
from wde.workflows.decoding.kv_cache.prefix_caching.allocator import (
    Block, PrefixCachingBlockAllocator, PrefixCachingVirtualBlockTable)
from wde.workflows.decoding.kv_cache.prefix_caching.lru_evictor import \
    LRUEvictor
from wde.workflows.decoding.kv_cache.prefix_caching.util import (PrefixHash,
                                                                 TokenIDs,
                                                                 get_block_hash
                                                                 )
from wde.workflows.decoding.kv_cache.utils import chunk_list
from wde.workflows.decoding.kv_cache.yoco.copy_on_write import CopyOnWrite
from wde.workflows.decoding.kv_cache.yoco.trie import Trie

logger = init_logger(__name__)


class YOCOVirtualBlockTable(PrefixCachingVirtualBlockTable):
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
        super().__init__(block_size=block_size,
                         block_allocator=block_allocator,
                         init_prefix_hash=init_prefix_hash,
                         _blocks=_blocks)
        # wait to write & cow
        self._wait = False

    #####################################
    # lock
    # Reading kv cache does not require lock
    # Writing to kv cache require lock
    # release lock when update_num_computed_tokens

    def ready(self) -> bool:
        # wait to write & cow
        if self._wait:
            return False

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
            # _num_token_ids update in _update
            self._update(token_ids)

        self._update_num_computed_tokens()

    def update_num_computed_tokens(self):
        seq_len = self.seq_len
        block_size = self._block_size
        end = self._tail + 1 if self._head == self._tail else self._tail

        for i in range(self._head, end):
            block = self._blocks[i]

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
    # helper function

    def _update(self, token_ids: TokenIDs):
        num_token_ids = len(token_ids)
        num_token_ids_curr = self.num_token_ids
        assert num_token_ids >= num_token_ids_curr

        block_size = self._block_size
        block_allocator = self._allocator

        self._wait = False

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

        self._num_token_ids = offset

        if delta_token_ids != last_block.delta_token_ids:
            new_last_block, num_tokens = block_allocator.get_portion_block(
                prefix_hash, delta_token_ids)

            self._num_token_ids += num_tokens

            if new_last_block is not last_block:
                block_allocator.hold(new_last_block)
                block_allocator.free(last_block)

            self._blocks[-1] = new_last_block

            if num_tokens < len(delta_token_ids):
                # wait at this block, until being computed
                self._wait = True
                return
        else:
            self._num_token_ids += len(delta_token_ids)

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

        self._wait = False

        for delta_token_ids in chunk_list(token_ids, block_size):
            if self._wait:
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
                        self._wait = True
                prefix_hash = block_hash
            else:
                # last portion block
                block, num_tokens = block_allocator.get_portion_block(
                    prefix_hash, delta_token_ids)

            self._num_token_ids += num_tokens

            block_allocator.hold(block)
            self._blocks.append(block)


class YOCOPrefixCachingBlockAllocator(PrefixCachingBlockAllocator):

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        model_name: str,
        kv_cache: List[torch.tensor],
        block_ids: Optional[Iterable[int]] = None,
    ):
        super().__init__(num_blocks=num_blocks,
                         block_size=block_size,
                         model_name=model_name,
                         block_ids=block_ids)

        self._portion_blocks_tries: Dict[PrefixHash, Trie] = {}
        self._free_blocks = LRUEvictor()
        self._cow_thread = CopyOnWrite(kv_cache=kv_cache, block_allocator=self)

    def create(self):
        return YOCOVirtualBlockTable(block_size=self._block_size,
                                     block_allocator=self,
                                     init_prefix_hash=self._init_prefix_hash)

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids) + len(self._free_blocks)

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
