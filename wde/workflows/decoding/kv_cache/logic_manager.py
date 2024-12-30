from wde.workflows.decoding.schema.request import DecodingSchedulableRequest

BlockId = int


class NoFreeBlocksError(ValueError):
    pass


class LogicKVCacheManager:

    def __init__(self, engine_config, block_allocator_class):
        self.engine_config = engine_config
        num_gpu_blocks = self.engine_config.cache_config.num_gpu_blocks
        self._block_size = self.engine_config.cache_config.block_size
        self.block_allocator = block_allocator_class(
            num_blocks=num_gpu_blocks, block_size=self._block_size)

        watermark = self.engine_config.cache_config.watermark
        self.watermark_blocks = int(watermark * num_gpu_blocks)

    @classmethod
    def from_engine(cls, engine):
        return cls(engine_config=engine.engine_config)

    @property
    def num_free_blocks(self):
        return self.block_allocator.num_free_blocks

    def high_watermark(self) -> bool:
        return self.block_allocator.num_free_blocks < self.watermark_blocks

    def create_vblock(self, request: DecodingSchedulableRequest):
        request.vblock = self.block_allocator.create_vblock()

    def update(self, request: DecodingSchedulableRequest):
        token_ids = request.get_token_ids()
        request.vblock.update(token_ids)

    def can_allocate(self, request: DecodingSchedulableRequest,
                     budget_bound_token_chunk_size: int) -> int:
        return request.vblock.can_allocate(budget_bound_token_chunk_size)

    def allocate(self, request: DecodingSchedulableRequest) -> None:
        request.vblock.allocate(request.token_chunk_size)
        assert request.vblock.seq_len == request.vblock.num_computed_tokens + request.token_chunk_size

    def free(self, request: DecodingSchedulableRequest) -> None:
        request.vblock.free()

    def free_last_block(self, request: DecodingSchedulableRequest):
        request.vblock.free_last_block()
        request.num_preempted += 1


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
