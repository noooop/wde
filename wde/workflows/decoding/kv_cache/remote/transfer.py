from typing import TYPE_CHECKING, List

import numpy as np

from wde.workflows.decoding.kv_cache.remote.memory import get_share_memory_np

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.offloading.manager import (
        CPUBlock, CPUBlockAllocator)
    from wde.workflows.decoding.kv_cache.offloading.swap import SwapTask
    from wde.workflows.decoding.kv_cache.remote.manager import RemoteManager


class TransferTask:

    def __init__(self, blocks: List["CPUBlock"],
                 transfer_manager: "TransferOutManager"):
        self.transfer_manager = transfer_manager
        self.blocks = blocks
        self.future = None

        cpu_block_allocator = self.transfer_manager.cpu_block_allocator
        for cpu_block in blocks:
            cpu_block_allocator.hold(cpu_block)

    @classmethod
    def from_swap_out_task(cls, swap_out_task: "SwapTask",
                           transfer_manager: "TransferOutManager"):
        blocks = []
        for gpu_block, cpu_block in swap_out_task.need_swap:
            blocks.append(cpu_block)

        transfer_task = cls(blocks=blocks, transfer_manager=transfer_manager)
        swap_out_task.add_callback(transfer_task.submit)

        return transfer_task

    def submit(self):
        self.future = self.transfer_manager.submit(self)

    def wait(self):
        if self.future is None:
            return
        self.future.result()

    def do_callback(self):
        self.transfer_manager.finish_callback(self.blocks)


class TransferOutManager:

    def __init__(self, server_name, name, cpu_cache,
                 remote_manager: "RemoteManager",
                 cpu_block_allocator: "CPUBlockAllocator"):
        self.name = name
        self.server_name = server_name
        self.cpu_cache = get_share_memory_np(cpu_cache)

        self.remote_manager = remote_manager
        self.cpu_block_allocator = cpu_block_allocator

    def transfer(self, task: TransferTask):
        from wde.workflows.decoding.kv_cache.remote.client import \
            ZeroRemoteKVCacheClient
        client = ZeroRemoteKVCacheClient()

        try:
            block_hashs = []
            hash2block = {}
            for block in task.blocks:
                block_hashs.append(block.block_hash)
                hash2block[block.block_hash] = block

            block_hashs = np.array(block_hashs, dtype=np.int64)

            response = client.contains(self.server_name, self.name,
                                       block_hashs)

            need_to_transfer_block_hashs = []
            need_to_transfer_blocks = []
            for block_hash in response.miss:
                block = hash2block[block_hash]
                need_to_transfer_block_hashs.append(block_hash)
                need_to_transfer_blocks.append(
                    self.cpu_cache[block.physical_block_id])

            need_to_transfer_block_hashs = np.array(
                need_to_transfer_block_hashs, dtype=np.int64)
            client.set(self.server_name, self.name,
                       need_to_transfer_block_hashs, need_to_transfer_blocks)
        except Exception:
            import traceback
            traceback.print_exc()

        finally:
            self.remote_manager.finishd_task.put(task)

    def submit(self, task: TransferTask):
        return self.remote_manager.threads.submit(self.transfer, task)

    def finish_callback(self, blocks: List["CPUBlock"]):
        for cpu_block in blocks:
            self.cpu_block_allocator.free(cpu_block)
