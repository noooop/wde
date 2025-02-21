from typing import TYPE_CHECKING, List

BlockId = int

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.prefix_caching.allocator import Block
    from wde.workflows.decoding.schema.request import \
        DecodingSchedulableRequest


class NoFreeBlocksError(ValueError):
    pass


class LogicKVCacheManager:

    def __init__(self, engine_config, block_allocator):
        self.engine_config = engine_config
        self._num_gpu_blocks = engine_config.cache_config.num_gpu_blocks

        self._block_size = self.engine_config.cache_config.block_size
        self.block_allocator = block_allocator

        watermark = self.engine_config.cache_config.watermark
        self.watermark_blocks = int(watermark * self._num_gpu_blocks)

    @property
    def name(self):
        return self.block_allocator.__class__.__name__

    @classmethod
    def from_engine(cls, engine, block_allocator_class):
        kv_cache = engine.model_inputs_builder.kv_caches

        num_gpu_blocks = engine.engine_config.cache_config.num_gpu_blocks
        block_size = engine.engine_config.cache_config.block_size
        model_name = engine.engine_config.model_config.model

        block_allocator = block_allocator_class(num_blocks=num_gpu_blocks,
                                                block_size=block_size,
                                                kv_cache=kv_cache,
                                                model_name=model_name)

        return cls(engine_config=engine.engine_config,
                   block_allocator=block_allocator)

    @property
    def num_free_blocks(self):
        return self.block_allocator.num_free_blocks

    def high_watermark(self) -> bool:
        return self.block_allocator.num_free_blocks < self.watermark_blocks

    def create(self, request: "DecodingSchedulableRequest"):
        if request.vblock is not None:
            return

        request.vblock = self.block_allocator.create()

    def update(self, request: "DecodingSchedulableRequest"):
        token_ids = request.get_token_ids()
        request.vblock.update(token_ids)

    def can_allocate(self, request: "DecodingSchedulableRequest",
                     budget_bound_token_chunk_size: int) -> int:
        return request.vblock.can_allocate(budget_bound_token_chunk_size)

    def allocate(self, request: "DecodingSchedulableRequest") -> None:
        request.vblock.allocate(request.token_chunk_size)
        assert request.vblock.seq_len == request.vblock.num_computed_tokens + request.token_chunk_size

    def free(self, request: "DecodingSchedulableRequest") -> None:
        request.vblock.free()

    def free_last_block(self, request: "DecodingSchedulableRequest"):
        request.vblock.free_last_block()
        request.num_preempted += 1

    def join(self):
        self.block_allocator.join()


class VirtualBlockTableInterface:

    @property
    def seq_len(self):
        raise NotImplementedError

    @property
    def num_new_tokens(self):
        raise NotImplementedError

    @property
    def num_computed_tokens(self):
        raise NotImplementedError

    @property
    def physical_block_ids(self):
        raise NotImplementedError

    def can_allocate(self, token_chunk_size: int):
        raise NotImplementedError

    def allocate(self, token_chunk_size: int):
        raise NotImplementedError

    def free(self):
        raise NotImplementedError

    def free_last_block(self):
        raise NotImplementedError

    def update_num_computed_tokens(self):
        raise NotImplementedError

    ##################################
    # api for PrefixCaching

    def update(self, token_ids: List[int]):
        raise NotImplementedError


class BlockAllocatorInterface:

    def create(self):
        raise NotImplementedError

    @property
    def num_free_blocks(self) -> int:
        raise NotImplementedError

    def allocate(self, *args, **kwargs):
        raise NotImplementedError

    def hold(self, *args, **kwargs):
        raise NotImplementedError

    def free(self, *args, **kwargs):
        raise NotImplementedError

    def join(self):
        pass

    ##################################
    # api for PrefixCaching

    def update(self, block: "Block"):
        return block
