from collections import deque
from typing import Deque, Iterable, Optional

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.interfaces import (BlockAllocator,
                                                        BlockId,
                                                        NoFreeBlocksError)
from wde.workflows.decoding.kv_cache.manager import AllocStatus, KVCacheManager
from wde.workflows.decoding.kv_cache.utils import RefCounter
from wde.workflows.decoding.kv_cache.virtual_table import (
    Block, VirtualBlockTable, get_num_required_blocks)
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

logger = init_logger(__name__)

RequestId = str


class NaiveKVCacheManager(KVCacheManager):

    def __init__(self, *args, watermark: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)

        num_gpu_blocks = self.engine_config.cache_config.num_gpu_blocks
        num_cpu_blocks = self.engine_config.cache_config.num_cpu_blocks

        self._block_size = self.engine_config.cache_config.block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_allocator = NaiveBlockAllocator(num_blocks=num_gpu_blocks,
                                                   block_size=self._block_size)

        self.watermark = watermark
        assert watermark >= 0.0
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        sliding_window = self.engine_config.cache_config.sliding_window

        self.sliding_window = sliding_window
        self.max_block_sliding_window = None
        if sliding_window is not None:
            num_blocks = sliding_window // self._block_size + 1
            self.max_block_sliding_window = num_blocks + 1

    def can_allocate(self, request: DecodingSchedulableRequest) -> AllocStatus:
        num_required_blocks = get_num_required_blocks(
            request.get_token_ids(), block_size=self._block_size)

        num_free_gpu_blocks = self.block_allocator.num_free_blocks

        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, request: DecodingSchedulableRequest) -> None:
        vblock = VirtualBlockTable(
            block_size=self._block_size,
            block_allocator=self.block_allocator,
        )
        token_ids = request.get_token_ids()

        if token_ids:
            vblock.allocate(token_ids)

        request.vblock = vblock

    def can_append_slots(self, request: DecodingSchedulableRequest) -> bool:
        vblock = request.vblock
        token_ids = request.get_token_ids()

        num_required_blocks = get_num_required_blocks(token_ids,
                                                      self._block_size)
        num_new_blocks_needed = num_required_blocks - len(vblock)

        num_free_gpu_blocks = self.block_allocator.num_free_blocks

        return num_new_blocks_needed <= num_free_gpu_blocks

    def append_slots(
        self,
        request: DecodingSchedulableRequest,
    ):
        token_ids = request.get_token_ids()
        request.vblock.append_token_ids(token_ids)

    def free(self, request: DecodingSchedulableRequest) -> None:
        request.vblock.free()


class NaiveBlock(Block):
    pass


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
        self._physical_block_refcounter = RefCounter(
            num_total_blocks=self.num_total_blocks)

    @property
    def num_total_blocks(self) -> int:
        return self._num_blocks

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_physical_block_ids)

    def allocate_block(self, prev_block: Optional[Block]):
        block_id = self._get_free_physical_block_id()

        block = NaiveBlock(prev_block=prev_block,
                           block_size=self._block_size,
                           block_id=block_id)

        return block

    def free(self, block: Block) -> None:
        self._put_physical_block_id(block)

    def _get_free_physical_block_id(self):
        if not self._free_physical_block_ids:
            raise NoFreeBlocksError()

        block_id = self._free_physical_block_ids.popleft()
        self._physical_block_refcounter.incr(block_id)
        return block_id

    def _put_physical_block_id(self, block: Block) -> None:
        block_id = block.block_id
        assert block_id is not None

        refcount = self._physical_block_refcounter.decr(block_id)
        if refcount == 0:
            self._free_physical_block_ids.appendleft(block_id)
