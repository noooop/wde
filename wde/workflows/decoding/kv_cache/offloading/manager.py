import queue
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Deque, Dict, Optional

from wde.workflows.core.executor.stream_pool import StreamPool
from wde.workflows.decoding.kv_cache.logic_manager import (BlockId,
                                                           NoFreeBlocksError,
                                                           PrefixHash)
from wde.workflows.decoding.kv_cache.offloading.swap import (SwapInManager,
                                                             SwapOutManager)
from wde.workflows.decoding.kv_cache.prefix_caching.lru_evictor import \
    LRUEvictor
from wde.workflows.decoding.schema.engine_io import DecodingSchedulerOutput


@dataclass
class CPUBlock:
    block_hash: Optional[PrefixHash] = None
    physical_block_id: Optional[BlockId] = None
    ref_count: int = 0
    lock: bool = False

    def incr(self):
        self.ref_count += 1
        return self.ref_count

    def decr(self):
        assert self.ref_count > 0
        self.ref_count -= 1
        return self.ref_count

    def ready(self):
        return not self.lock

    def acquire(self):
        if self.ready():
            self.lock = True
            return True

        return False

    def release(self):
        assert self.lock
        self.lock = False


class CPUBlockAllocator:

    def __init__(self, num_blocks, block_size):
        self.num_blocks = num_blocks
        self._block_size = block_size

        self._num_blocks = num_blocks
        self._full_blocks_map: Dict[PrefixHash, CPUBlock] = {}

        self._free_full_blocks = LRUEvictor()
        self._free_physical_block_ids: Deque[BlockId] = deque(
            range(num_blocks))

    def __contains__(self, block_hash):
        return block_hash in self._full_blocks_map

    def __len__(self):
        return len(self._full_blocks_map)

    def get(self, block_hash):
        block = self._full_blocks_map.get(block_hash, None)

        if block is None:
            return None

        if not block.ready():
            return None

        self._free_full_blocks.remove(block)

        return block

    def create(self, block_hash):
        block = self._full_blocks_map.get(block_hash, None)

        if block is not None:
            return block

        try:
            physical_block_id = self._get_free_physical_block_id()
        except NoFreeBlocksError:
            return

        new_block = CPUBlock(block_hash=block_hash,
                             physical_block_id=physical_block_id,
                             lock=True)

        self._full_blocks_map[new_block.block_hash] = new_block

        return new_block

    def copy_block(self, block):
        assert block.block_hash is not None

        try:
            physical_block_id = self._get_free_physical_block_id()
        except NoFreeBlocksError:
            return

        new_block = CPUBlock(block_hash=block.block_hash,
                             physical_block_id=physical_block_id)

        self._full_blocks_map[new_block.block_hash] = new_block
        return new_block

    def hold(self, block: CPUBlock):
        ref_count = block.incr()

        if ref_count == 1:
            self._remove_from_free_blocks(block)

    def free(self, block: CPUBlock) -> None:
        ref_count = block.decr()

        if ref_count == 0:
            assert self._full_blocks_map[block.block_hash] is block
            self._free_full_blocks.add(block)

    def refresh(self, block: CPUBlock):
        self._free_full_blocks.update(block)

    def _remove_from_free_blocks(self, block: CPUBlock):
        if block is None:
            return

        self._free_full_blocks.remove(block)

    def _get_free_physical_block_id(self):
        try:
            physical_block_id = self._free_physical_block_ids.popleft()
            return physical_block_id
        except IndexError:
            pass

        full_blocks = self._free_full_blocks.evict()
        self._full_blocks_map.pop(full_blocks.block_hash, None)
        return full_blocks.physical_block_id


class OffloadingManager:

    def __init__(self, engine_config, cpu_cache, gpu_cache,
                 gpu_block_allocator):
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
            offloading_manager=self)

        self.swap_in_manager = SwapInManager(
            cpu_cache,
            gpu_cache,
            gpu_block_allocator=self.gpu_block_allocator,
            cpu_block_allocator=self.cpu_block_allocator,
            offloading_manager=self,
        )

        self.stream_pool = StreamPool()
        self.threads = ThreadPoolExecutor(
            max_workers=self.engine_config.sys_config.
            kv_cache_offloading_max_workers)
        self.finishd_task = queue.Queue()

    def get_swap_out_task(self, scheduler_outputs):
        return self.swap_out_manager.prepare_task(scheduler_outputs)

    def get_swap_in_blocks(self, request):
        return self.swap_in_manager.get_swap_in_blocks(request)

    def get_swap_in_task(self, scheduler_outputs: DecodingSchedulerOutput):
        swap_in_task = self.swap_in_manager.get_swap_in_task(
            scheduler_outputs.need_swap_in_blocks)
        scheduler_outputs.need_swap_in_blocks = None

        if swap_in_task is not None:
            swap_in_task.submit()

        return swap_in_task

    def try_allocate_swap_in_blocks(self, need_swap_in_blocks):
        return self.swap_in_manager.try_allocate_swap_in_blocks(
            need_swap_in_blocks)

    def check_finishd_task(self):
        while True:
            try:
                task = self.finishd_task.get(block=False)
            except queue.Empty:
                break
            task.do_callback()
