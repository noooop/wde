import queue
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

import torch
from vllm import _custom_ops as ops

from wde.workflows.core.executor.stream_pool import StreamPool

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.offloading.manager import \
        CPUBlockAllocator


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


class SwapOutTask:

    def __init__(self, need_swap_out, swap_out_manager: "SwapOutManager",
                 stream: torch.cuda.Stream):
        self.swap_out_manager = swap_out_manager
        self.need_swap_out = need_swap_out
        self.stream = stream

    def swap_blocks(self):
        num_attention_layers = self.swap_out_manager.num_attention_layers
        gpu_cache = self.swap_out_manager.gpu_cache
        cpu_cache = self.swap_out_manager.cpu_cache

        block_mapping = [(b1.physical_block_id, b2.physical_block_id)
                         for b1, b2 in self.need_swap_out]

        blocks_to_swap = torch.tensor(block_mapping,
                                      device="cpu",
                                      dtype=torch.int64).view(-1, 2)

        for i in range(num_attention_layers):
            swap_layer(gpu_cache[i], cpu_cache[i], blocks_to_swap)

    def swap_out(self):
        with torch.cuda.stream(self.stream):
            self.swap_blocks()
        self.stream.synchronize()
        self.swap_out_manager.finishd_task.put(self)
        self.swap_out_manager.stream_pool.put(self.stream)

    def do_callback(self):
        self.swap_out_manager.finish_callback(self.need_swap_out)

    def submit(self):
        self.swap_out_manager.threads.submit(self.swap_out)


class SwapOutManager:

    def __init__(self, cpu_cache, gpu_cache, gpu_block_allocator,
                 cpu_block_allocator: "CPUBlockAllocator", max_workers: int):
        assert cpu_cache is not None
        assert gpu_cache is not None
        assert len(cpu_cache) == len(gpu_cache)

        self.gpu_cache = gpu_cache
        self.cpu_cache = cpu_cache
        self.num_attention_layers = len(self.gpu_cache)

        self.stream_pool = StreamPool()
        self.gpu_block_allocator = gpu_block_allocator
        self.cpu_block_allocator = cpu_block_allocator
        self.threads = ThreadPoolExecutor(max_workers=max_workers)
        self.finishd_task = queue.Queue()

    def prepare_task(self, scheduler_outputs) -> Optional[SwapOutTask]:
        remove_duplicates = set()

        need_swap_out = []

        for request in scheduler_outputs.scheduled_requests:
            new_full_blocks = request.vblock.new_full_blocks()

            for gpu_block in new_full_blocks:
                self_prefix_hash = gpu_block.self_prefix_hash

                if self_prefix_hash in self.cpu_block_allocator:
                    continue

                if self_prefix_hash in remove_duplicates:
                    continue

                remove_duplicates.add(self_prefix_hash)

                cpu_block = self.cpu_block_allocator.copy_block(gpu_block)

                if cpu_block is None:
                    break

                gpu_block.incr()
                cpu_block.incr()

                need_swap_out.append((gpu_block, cpu_block))

        if not need_swap_out:
            return None

        stream = self.stream_pool.get()

        return SwapOutTask(need_swap_out=need_swap_out,
                           swap_out_manager=self,
                           stream=stream)

    def finish_callback(self, need_swap_out):
        for gpu_block, cpu_block in need_swap_out:
            self.gpu_block_allocator.free(gpu_block)
            self.cpu_block_allocator.free(cpu_block)

    def check_finishd_task(self):
        while True:
            try:
                task = self.finishd_task.get(block=False)
            except queue.Empty:
                break

            task.do_callback()
