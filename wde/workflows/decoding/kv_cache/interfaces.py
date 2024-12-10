from dataclasses import dataclass
from typing import Optional

BlockId = int


@dataclass
class Block:
    prev_block: Optional["Block"]
    block_size: int
    block_id: BlockId


class NoFreeBlocksError(ValueError):
    pass


class BlockAllocator:

    @property
    def num_total_blocks(self) -> int:
        raise NotImplementedError

    @property
    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def allocate_block(self, prev_block: Optional[Block]):
        raise NotImplementedError

    def free(self, block: Block) -> None:
        raise NotImplementedError
