"""A block manager that manages token blocks."""
import enum
from typing import Dict, List, Optional, Tuple

from vllm.core.block.block_table import BlockTable
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.utils import Device

from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

RequestId = str


class AllocStatus(enum.Enum):
    """Result for BlockSpaceManager.can_allocate

    1. Ok: seq_group can be allocated now.
    2. Later: seq_group cannot be allocated.
      The capacity of allocator is larger than seq_group required.
    3. Never: seq_group can never be allocated.
      The seq_group is too large to allocated in GPU.
    """
    OK = enum.auto()
    LATER = enum.auto()
    NEVER = enum.auto()


class SelfAttnBlockSpaceManager:

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks

        self.sliding_window = sliding_window
        # max_block_sliding_window is the max number of blocks that need to be
        # allocated
        self.max_block_sliding_window = None
        if sliding_window is not None:
            # +1 here because // rounds down
            num_blocks = sliding_window // block_size + 1
            # +1 here because the last block may not be full,
            # and so the sequence stretches one more block at the beginning
            # For example, if sliding_window is 3 and block_size is 4,
            # we may need 2 blocks when the second block only holds 1 token.
            self.max_block_sliding_window = num_blocks + 1

        self.watermark = watermark
        assert watermark >= 0.0

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)

        self.block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.block_tables: Dict[RequestId, BlockTable] = {}

    def can_allocate(self,
                     request: DecodingSchedulableRequest,
                     num_lookahead_slots: int = 0) -> AllocStatus:
        num_required_blocks = BlockTable.get_num_required_blocks(
            request.get_token_ids(),
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )

        if self.max_block_sliding_window is not None:
            num_required_blocks = min(num_required_blocks,
                                      self.max_block_sliding_window)

        num_free_gpu_blocks = self.block_allocator.get_num_free_blocks(
            device=Device.GPU)

        # Use watermark to avoid frequent cache eviction.
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
            # Add blocks to the block table only if the sequence is non empty.
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
