from typing import TYPE_CHECKING, Optional

import torch
from vllm import _custom_ops as ops

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.offloading.manager import (
        CPUBlockAllocator, OffloadingManager)
    from wde.workflows.decoding.kv_cache.prefix_caching.allocator import \
        PrefixCachingBlockAllocator


def swap_layer(
    src_kv_cache: torch.Tensor,
    dst_kv_cache: torch.Tensor,
    src_to_dst: torch.Tensor,
) -> None:
    src_key_cache = src_kv_cache[0]
    dst_key_cache = dst_kv_cache[0]
    ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
    src_value_cache = src_kv_cache[1]
    dst_value_cache = dst_kv_cache[1]
    ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)


class SwapTask:

    def __init__(self, swap_manager, need_swap=None):
        self.swap_manager = swap_manager
        self.need_swap = need_swap or []
        self.future = None

    def add(self, items):
        self.need_swap.extend(items)

    def swap_blocks(self):
        num_attention_layers = self.swap_manager.num_attention_layers
        from_cache = self.swap_manager.from_cache
        to_cache = self.swap_manager.to_cache

        block_mapping = [(b1.physical_block_id, b2.physical_block_id)
                         for b1, b2 in self.need_swap]

        blocks_to_swap = torch.tensor(block_mapping,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        for i in range(num_attention_layers):
            swap_layer(from_cache[i], to_cache[i], blocks_to_swap)

    def swap(self):
        stream = self.swap_manager.offloading_manager.stream_pool.get()
        with torch.cuda.stream(stream):
            self.swap_blocks()
        stream.synchronize()
        self.swap_manager.offloading_manager.stream_pool.put(stream)
        self.swap_manager.offloading_manager.finishd_task.put(self)

    def submit(self):
        self.future = self.swap_manager.offloading_manager.threads.submit(
            self.swap)

    def wait(self):
        if self.future is None:
            return
        self.future.result()

    def do_callback(self):
        self.swap_manager.finish_callback(self.need_swap)


class SwapOutManager:

    def __init__(self, cpu_cache, gpu_cache,
                 gpu_block_allocator: "PrefixCachingBlockAllocator",
                 cpu_block_allocator: "CPUBlockAllocator",
                 offloading_manager: "OffloadingManager"):
        assert cpu_cache is not None
        assert gpu_cache is not None
        assert len(cpu_cache) == len(gpu_cache)

        self.from_cache = gpu_cache
        self.to_cache = cpu_cache
        self.num_attention_layers = len(gpu_cache)

        self.gpu_block_allocator = gpu_block_allocator
        self.cpu_block_allocator = cpu_block_allocator
        self.offloading_manager = offloading_manager

    def prepare_task(self, scheduler_outputs) -> Optional[SwapTask]:
        remove_duplicates = set()

        need_swap_out = []

        for request in scheduler_outputs.scheduled_requests:
            new_full_blocks = request.vblock.new_full_blocks()

            for gpu_block in new_full_blocks:
                gpu_block.ensure_self_prefix_hash()

                self_prefix_hash = gpu_block.self_prefix_hash

                if self_prefix_hash in self.cpu_block_allocator:
                    continue

                if self_prefix_hash in remove_duplicates:
                    continue

                remove_duplicates.add(self_prefix_hash)

                cpu_block = self.cpu_block_allocator.copy_block(gpu_block)

                if cpu_block is None:
                    break

                # read from gpu_block
                gpu_block.incr()

                # write to cpu_block
                cpu_block.incr()
                cpu_block.acquire()

                need_swap_out.append((gpu_block, cpu_block))

        if not need_swap_out:
            return None

        return SwapTask(need_swap=need_swap_out, swap_manager=self)

    def finish_callback(self, need_swap_out):
        for gpu_block, cpu_block in need_swap_out:
            cpu_block.release()
            self.gpu_block_allocator.free(gpu_block)
            self.cpu_block_allocator.free(cpu_block)


class SwapInManager:

    def __init__(self, cpu_cache, gpu_cache,
                 gpu_block_allocator: "PrefixCachingBlockAllocator",
                 cpu_block_allocator: "CPUBlockAllocator",
                 offloading_manager: "OffloadingManager"):
        assert cpu_cache is not None
        assert gpu_cache is not None
        assert len(cpu_cache) == len(gpu_cache)

        self.from_cache = cpu_cache
        self.to_cache = gpu_cache
        self.num_attention_layers = len(gpu_cache)
        self.gpu_block_allocator = gpu_block_allocator
        self.cpu_block_allocator = cpu_block_allocator
        self.offloading_manager = offloading_manager

    def get_swap_in_task(self, need_swap_in_blocks):
        if not need_swap_in_blocks:
            return None
        return SwapTask(swap_manager=self, need_swap=need_swap_in_blocks)

    def get_swap_in_blocks(self, request):
        assert request.vblock is not None

        maybe_swap_in_blocks = request.vblock.get_maybe_swap_in_blocks()
        need_swap_in_blocks = []

        for gpu_block in maybe_swap_in_blocks:
            if gpu_block.lock:
                continue

            gpu_block.ensure_self_prefix_hash()

            self_prefix_hash = gpu_block.self_prefix_hash

            cpu_block = self.cpu_block_allocator.get(self_prefix_hash)

            if cpu_block is not None:
                need_swap_in_blocks.append((cpu_block, gpu_block))

        return need_swap_in_blocks

    def finish_callback(self, need_swap_out):
        for cpu_block, gpu_block in need_swap_out:
            gpu_block.num_computed_tokens = gpu_block._block_size
            gpu_block.release()
            self.cpu_block_allocator.free(cpu_block)
