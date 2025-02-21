from typing import TYPE_CHECKING, List, Tuple

import torch
from vllm import _custom_ops as ops

from wde.workflows.decoding.kv_cache.prefix_caching.allocator import Block

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.yoco.allocator import \
        YOCOPrefixCachingBlockAllocator


class CopyOnWrite:

    def __init__(self, kv_cache: List[torch.Tensor],
                 block_allocator: "YOCOPrefixCachingBlockAllocator"):
        self.task: List[Tuple[Block, Block]] = []
        self.stream = torch.cuda.Stream()
        self.key_caches = [kv_cache[0] for kv_cache in kv_cache]
        self.value_caches = [kv_cache[1] for kv_cache in kv_cache]
        self.block_allocator = block_allocator

    def submit(self, from_: Block, to_: Block, n_token):
        from_physical_block_id = from_.physical_block_id
        to_physical_block_id = to_.physical_block_id

        assert from_physical_block_id is not None
        assert to_physical_block_id is not None

        to_.acquire()
        self.block_allocator.hold(from_)
        self.block_allocator.hold(to_)

        with torch.cuda.stream(self.stream):
            self.ops_copy_blocks(from_physical_block_id, to_physical_block_id,
                                 n_token)

        self.task.append((from_, to_))

    def ops_copy_blocks(self, from_physical_block_id, to_physical_block_id,
                        n_token):
        blocks_to_copy = torch.tensor(
            [(from_physical_block_id, to_physical_block_id)],
            device="cuda",
            dtype=torch.int64).view(-1, 2)

        ops.copy_blocks(self.key_caches, self.value_caches, blocks_to_copy)

    def torch_copy_blocks(self, from_physical_block_id, to_physical_block_id,
                          n_token):
        for key in self.key_caches:
            key[to_physical_block_id][:n_token].copy_(
                key[from_physical_block_id][:n_token], non_blocking=True)

        for value in self.value_caches:
            value[to_physical_block_id][:n_token].copy_(
                value[from_physical_block_id][:n_token], non_blocking=True)

    def join(self):
        self.stream.synchronize()

        for from_, to_ in self.task:
            to_.release()
            self.block_allocator.free(from_)
            self.block_allocator.free(to_)

        self.task = []
