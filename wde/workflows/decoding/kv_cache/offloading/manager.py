import queue
import time
from collections import deque
from queue import PriorityQueue
from typing import Deque, Dict

from wde.workflows.decoding.kv_cache.logic_manager import (BlockId,
                                                           NoFreeBlocksError,
                                                           PrefixHash)
from wde.workflows.decoding.kv_cache.offloading.swap_out import SwapOutManager
from wde.workflows.decoding.kv_cache.prefix_caching.manager import (
    Block, PrefixCachingBlockAllocator)


class CPUBlockAllocator:

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self._block_size = block_size

        self._full_blocks_map: Dict[PrefixHash, Block] = {}
        self._free_full_blocks: PriorityQueue[Block] = PriorityQueue()

        block_ids = range(num_blocks)
        self._free_physical_block_ids: Deque[BlockId] = deque(block_ids)

    def __contains__(self, self_prefix_hash):
        return self_prefix_hash in self._full_blocks_map

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            pass

        try:
            full_blocks = self._free_full_blocks.get()
            del self._full_blocks_map[full_blocks.self_prefix_hash]
            return full_blocks.physical_block_id
        except queue.Empty:
            raise NoFreeBlocksError()

    def free(self, block: Block) -> None:
        ref_count = block.decr()

        if ref_count == 0:
            block.last_accessed_ts = time.time()
            self._free_full_blocks.put(block)

    def copy_block(self, block):
        try:
            physical_block_id = self._get_free_physical_block_id()
        except NoFreeBlocksError:
            return

        new_block = Block(delta_token_ids=block.delta_token_ids,
                          prefix_hash=block.prefix_hash,
                          self_prefix_hash=block.self_prefix_hash,
                          _block_size=block._block_size,
                          physical_block_id=physical_block_id)

        return new_block


class OffloadingManager:

    def __init__(self, engine_config, cpu_cache, gpu_cache,
                 gpu_block_allocator: PrefixCachingBlockAllocator):
        self.engine_config = engine_config
        self.gpu_block_allocator = gpu_block_allocator
        self.cpu_block_allocator = CPUBlockAllocator(
            num_blocks=engine_config.cache_config.num_cpu_blocks,
            block_size=engine_config.cache_config.block_size)
        self.swap_out_manager = SwapOutManager(
            cpu_cache,
            gpu_cache,
            gpu_block_allocator=self.gpu_block_allocator,
            cpu_block_allocator=self.cpu_block_allocator,
            max_workers=self.engine_config.scheduler_config.max_num_on_the_fly)

    def get_swap_out_task(self, scheduler_outputs):
        return self.swap_out_manager.prepare_task(scheduler_outputs)

    def check_swap_out_finishd_task(self):
        self.swap_out_manager.check_finishd_task()
