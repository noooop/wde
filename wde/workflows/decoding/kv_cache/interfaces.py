BlockId = int


class NoFreeBlocksError(ValueError):
    pass


class BlockAllocator:

    @property
    def num_total_blocks(self) -> int:
        raise NotImplementedError

    @property
    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def create_vblock(self):
        raise NotImplementedError

    def allocate_block(self):
        raise NotImplementedError

    def free(self, physical_block_id: BlockId) -> None:
        raise NotImplementedError


class VirtualBlockTable:

    @property
    def num_token_ids(self):
        raise NotImplementedError

    @property
    def num_computed_tokens(self):
        raise NotImplementedError

    @property
    def num_empty_slots(self):
        raise NotImplementedError

    @property
    def max_num_token_ids(self):
        raise NotImplementedError

    def allocate(self, token_ids):
        raise NotImplementedError

    def free(self):
        raise NotImplementedError

    def free_last_block(self):
        raise NotImplementedError

    @property
    def physical_block_ids(self):
        raise NotImplementedError

    def update_num_computed_tokens(self):
        raise NotImplementedError
