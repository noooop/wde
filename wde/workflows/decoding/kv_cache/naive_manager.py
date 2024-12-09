from typing import Dict, List, Tuple

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.utils import Device

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.manager import AllocStatus, KVCacheManager
from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

logger = init_logger(__name__)

RequestId = str


class NaiveKVCacheManager(KVCacheManager):

    def __init__(self, *args, watermark: float = 0.01, **kwargs):
        super().__init__(*args, **kwargs)

        num_gpu_blocks = self.engine_config.cache_config.num_gpu_blocks
        num_cpu_blocks = self.engine_config.cache_config.num_cpu_blocks

        self.block_size = self.engine_config.cache_config.block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=self.block_size,
        )

        self.watermark = watermark
        assert watermark >= 0.0
        self.watermark_blocks = int(watermark * num_gpu_blocks)

        sliding_window = self.engine_config.cache_config.sliding_window

        self.sliding_window = sliding_window
        self.max_block_sliding_window = None
        if sliding_window is not None:
            num_blocks = sliding_window // self.block_size + 1
            self.max_block_sliding_window = num_blocks + 1

        self.block_tables: Dict[RequestId, BlockTable] = {}

    def can_allocate(self,
                     request: DecodingSchedulableRequest,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        num_required_blocks = BlockTable.get_num_required_blocks(
            request.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        if (self.num_total_gpu_blocks - num_required_blocks
                < self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, request: DecodingSchedulableRequest) -> None:
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=self.block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
        )
        if request.get_token_ids():
            block_table.allocate(request.get_token_ids())

        self.block_tables[request.request_id] = block_table
        request.vblock = block_table

    def can_append_slots(self, request: DecodingSchedulableRequest,
                         num_lookahead_slots: int) -> bool:
        num_touched_blocks = 0

        block_table = request.vblock

        num_touched_blocks += (
            block_table.get_num_blocks_touched_by_append_slots(
                token_ids=block_table.get_unseen_token_ids(
                    request.get_token_ids()),
                num_lookahead_slots=num_lookahead_slots,
            ))

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        request: DecodingSchedulableRequest,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:

        block_table = request.vblock

        block_table.append_token_ids(
            token_ids=block_table.get_unseen_token_ids(
                request.get_token_ids()),
            num_lookahead_slots=num_lookahead_slots,
            num_computed_slots=request.get_num_computed_tokens(),
        )
        # Return any new copy-on-writes.
        new_cows = self.block_allocator.clear_copy_on_writes()
        return new_cows

    def free(self, request: DecodingSchedulableRequest) -> None:
        if request.request_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        request.vblock.free()

        # Free table/blocks
        self.block_tables[request.request_id].free()
        del self.block_tables[request.request_id]

    def get_block_table(self,
                        request: DecodingSchedulableRequest) -> List[int]:
        block_ids = self.block_tables[request.request_id].physical_block_ids
        return block_ids  # type: ignore

    def get_num_free_gpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.block_allocator.get_num_free_blocks(Device.CPU)
