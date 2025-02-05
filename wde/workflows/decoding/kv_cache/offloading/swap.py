from typing import TYPE_CHECKING, Optional

import torch
from vllm import _custom_ops as ops

from wde.workflows.decoding.kv_cache.logic_manager import NoFreeBlocksError

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.offloading.manager import (
        CPUBlockAllocator, OffloadingManager)
    from wde.workflows.decoding.kv_cache.prefix_caching.allocator import \
        PrefixCachingBlockAllocator


class SwapTask:

    def __init__(self, swap_manager, need_swap=None):
        self.swap_manager = swap_manager
        self.need_swap = need_swap or []
        self.future = None
        self.callbacks = []

    def add(self, items):
        self.need_swap.extend(items)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def swap_blocks(self):
        block_mapping = [(b1.physical_block_id, b2.physical_block_id)
                         for b1, b2 in self.need_swap]

        swap_blocks(from_cache=self.swap_manager.from_cache,
                    to_cache=self.swap_manager.to_cache,
                    block_mapping=block_mapping)

    def swap(self):
        stream = self.swap_manager.offloading_manager.stream_pool.get()

        try:
            with torch.cuda.stream(stream):
                self.swap_blocks()
            stream.synchronize()

            for from_block, to_block in self.need_swap:
                to_block.release()

            for callback in self.callbacks:
                callback()

        except Exception:
            import traceback
            traceback.print_exc()

        finally:
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

        self.from_cache = gpu_cache
        self.to_cache = cpu_cache

        self.gpu_block_allocator = gpu_block_allocator
        self.cpu_block_allocator = cpu_block_allocator
        self.offloading_manager = offloading_manager

    def prepare_task(self, scheduler_outputs) -> Optional[SwapTask]:
        remove_duplicates = set()

        need_swap_out = []

        for request in scheduler_outputs.scheduled_requests:
            new_full_blocks = request.vblock.new_full_blocks()

            for gpu_block in new_full_blocks:
                gpu_block.ensure_block_hash()

                block_hash = gpu_block.block_hash

                if block_hash in self.cpu_block_allocator:
                    continue

                if block_hash in remove_duplicates:
                    continue

                remove_duplicates.add(block_hash)

                cpu_block = self.cpu_block_allocator.copy_block(gpu_block)

                if cpu_block is None:
                    break

                # read from gpu_block
                self.gpu_block_allocator.hold(gpu_block)

                # write to cpu_block
                cpu_block.acquire()
                self.cpu_block_allocator.hold(cpu_block)

                need_swap_out.append((gpu_block, cpu_block))

        if not need_swap_out:
            return None

        return SwapTask(need_swap=need_swap_out, swap_manager=self)

    def finish_callback(self, need_swap_out):
        for gpu_block, cpu_block in need_swap_out:
            self.gpu_block_allocator.free(gpu_block)
            self.cpu_block_allocator.free(cpu_block)


class SwapInManager:

    def __init__(self, cpu_cache, gpu_cache,
                 gpu_block_allocator: "PrefixCachingBlockAllocator",
                 cpu_block_allocator: "CPUBlockAllocator",
                 offloading_manager: "OffloadingManager"):
        assert cpu_cache is not None
        assert gpu_cache is not None

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

            gpu_block.ensure_block_hash()

            block_hash = gpu_block.block_hash

            cpu_block = self.cpu_block_allocator.get(block_hash)

            if cpu_block is not None:
                need_swap_in_blocks.append((cpu_block, gpu_block))

        return need_swap_in_blocks

    def try_allocate_swap_in_blocks(self, need_swap_in_blocks):
        allocated_swap_in_blocks = []
        for cpu_block, gpu_block in need_swap_in_blocks:
            assert cpu_block.ready()
            # read from cpu_block
            self.cpu_block_allocator.hold(cpu_block)

            # write to gpu_block, need acquire lock
            assert gpu_block.ready()

            try:
                self.offloading_manager.gpu_block_allocator.allocate(gpu_block)
            except NoFreeBlocksError:
                break

            gpu_block.acquire()
            self.gpu_block_allocator.hold(gpu_block)

            allocated_swap_in_blocks.append((cpu_block, gpu_block))
        return allocated_swap_in_blocks

    def finish_callback(self, need_swap_out):
        for cpu_block, gpu_block in need_swap_out:
            gpu_block.num_computed_tokens = gpu_block._block_size
            self.cpu_block_allocator.free(cpu_block)
            self.gpu_block_allocator.free(gpu_block)


@torch.inference_mode
def swap_blocks(from_cache, to_cache, block_mapping):
    n = 2

    if isinstance(from_cache, list):
        # from gpu to cpu
        num_attention_layers = len(from_cache)
        index = 1
        blocks_to_swap = torch.tensor(block_mapping,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        blocks_to_swap[:, index] *= num_attention_layers * n

        shape = tuple(to_cache.shape)
        to_cache = to_cache.view((-1, ) + shape[3:])

        for i in range(num_attention_layers):
            ops.swap_blocks(from_cache[i][0], to_cache, blocks_to_swap.clone())

            blocks_to_swap[:, index] += 1

            ops.swap_blocks(from_cache[i][1], to_cache, blocks_to_swap.clone())

            blocks_to_swap[:, index] += 1

    else:
        # from cpu to gpu
        num_attention_layers = len(to_cache)
        index = 0
        blocks_to_swap = torch.tensor(block_mapping,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        blocks_to_swap[:, index] *= num_attention_layers * n

        shape = tuple(from_cache.shape)
        from_cache = from_cache.view((-1, ) + shape[3:])

        for i in range(num_attention_layers):
            ops.swap_blocks(from_cache, to_cache[i][0], blocks_to_swap.clone())

            blocks_to_swap[:, index] += 1

            ops.swap_blocks(from_cache, to_cache[i][1], blocks_to_swap.clone())

            blocks_to_swap[:, index] += 1
