from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np

from wde.workflows.decoding.kv_cache.remote.memory import get_share_memory_np

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.logic_manager import PrefixHash
    from wde.workflows.decoding.kv_cache.offloading.manager import (
        CPUBlock, CPUBlockAllocator)
    from wde.workflows.decoding.kv_cache.offloading.swap import SwapTask
    from wde.workflows.decoding.kv_cache.remote.manager import RemoteManager
    from wde.workflows.decoding.schema.request import \
        DecodingSchedulableRequest


class TaskBase:

    def __init__(self, transfer_manager: "BaseTransferManager"):
        self.transfer_manager = transfer_manager
        self.future = None

    def acquire(self):
        return NotImplemented

    def release(self):
        return NotImplemented

    def submit(self):
        self.acquire()
        self.future = self.transfer_manager.submit(self)

    def wait(self):
        if self.future is None:
            return
        self.future.result()

    def do_callback(self):
        self.release()


class BaseTransferManager:

    def __init__(self, server_name, name, cpu_cache,
                 remote_manager: "RemoteManager",
                 cpu_block_allocator: "CPUBlockAllocator"):
        self.name = name
        self.server_name = server_name
        self.cpu_cache = get_share_memory_np(cpu_cache)

        self.remote_manager = remote_manager
        self.cpu_block_allocator = cpu_block_allocator

    def transfer(self, task: TaskBase):
        raise NotImplementedError

    def submit(self, task: TaskBase):

        def exception_handling(task):
            try:
                self.transfer(task)
            except Exception as e:
                self.remote_manager.finishd_task.put(e)
                return

            self.remote_manager.finishd_task.put(task)

        return self.remote_manager.threads.submit(exception_handling, task)


class TransferOutTask(TaskBase):

    def __init__(self, blocks: List["CPUBlock"],
                 transfer_manager: "TransferOutManager"):
        super().__init__(transfer_manager)
        self.blocks = blocks

    def acquire(self):
        cpu_block_allocator = self.transfer_manager.cpu_block_allocator
        for cpu_block in self.blocks:
            cpu_block_allocator.hold(cpu_block)

    def release(self):
        cpu_block_allocator = self.transfer_manager.cpu_block_allocator
        for cpu_block in self.blocks:
            cpu_block_allocator.free(cpu_block)

    @classmethod
    def from_swap_out_task(cls, swap_out_task: "SwapTask",
                           transfer_manager: "TransferOutManager"):
        blocks = []
        for gpu_block, cpu_block in swap_out_task.need_swap:
            blocks.append(cpu_block)

        transfer_task = cls(blocks=blocks, transfer_manager=transfer_manager)
        swap_out_task.add_callback(transfer_task.submit)

        return transfer_task


class TransferOutManager(BaseTransferManager):

    def transfer(self, task: TransferOutTask):
        from wde.workflows.decoding.kv_cache.remote.client import \
            ZeroRemoteKVCacheClient
        client = ZeroRemoteKVCacheClient()

        block_hashs = []
        hash2block = {}
        for block in task.blocks:
            block_hashs.append(block.block_hash)
            hash2block[block.block_hash] = block

        block_hashs = np.array(block_hashs, dtype=np.int64)

        response = client.contains(self.server_name, self.name, block_hashs)

        need_to_transfer_block_hashs = []
        need_to_transfer_blocks = []
        for block_hash in response.miss:
            block = hash2block[block_hash]
            need_to_transfer_block_hashs.append(block_hash)
            need_to_transfer_blocks.append(
                self.cpu_cache[block.physical_block_id])

        need_to_transfer_block_hashs = np.array(need_to_transfer_block_hashs,
                                                dtype=np.int64)
        client.set(self.server_name, self.name, need_to_transfer_block_hashs,
                   need_to_transfer_blocks)


class TransferInTask(TaskBase):

    def __init__(self, blocks: Dict["PrefixHash", "CPUBlock"],
                 requests: List["DecodingSchedulableRequest"],
                 transfer_manager: "TransferInManager"):
        super().__init__(transfer_manager)
        self.blocks = blocks
        self.requests = requests
        self.hits = set()

    def acquire(self):
        cpu_block_allocator = self.transfer_manager.cpu_block_allocator
        for cpu_block in self.blocks.values():
            cpu_block.acquire()
            cpu_block_allocator.hold(cpu_block)

        for request in self.requests:
            request.busy = True

    def release(self):
        cpu_block_allocator = self.transfer_manager.cpu_block_allocator

        for block_hash, cpu_block in self.blocks.items():
            if block_hash in self.hits:
                cpu_block.release()
                cpu_block_allocator.free(cpu_block)
            else:
                cpu_block.lock = None
                cpu_block_allocator.free(cpu_block)

        for request in self.requests:
            request.busy = False


class TransferInManager(BaseTransferManager):

    def get_transfer_in_task(
        self, scheduled_requests: List["DecodingSchedulableRequest"]
    ) -> Optional[TransferInTask]:
        gpu_kv_cache_manager = self.remote_manager.gpu_kv_cache_manager
        cpu_block_allocator = self.cpu_block_allocator

        need_transfer_in_blocks = {}
        need_transfer_in_requests = []

        for request in scheduled_requests:
            gpu_kv_cache_manager.update(request)

            gpu_blocks = request.vblock.get_maybe_swap_in_blocks()

            if len(gpu_blocks) > 0:
                need_transfer_in_requests.append(request)

            for gpu_block in gpu_blocks:
                gpu_block.ensure_block_hash()

                block_hash = gpu_block.block_hash

                if block_hash in need_transfer_in_blocks:
                    continue

                cpu_block = cpu_block_allocator.create(block_hash)

                if cpu_block is None:
                    # NoFreeBlocksError
                    break

                if cpu_block.lock is not None:
                    # not newly created
                    continue

                cpu_block.acquire()
                cpu_block_allocator.hold(cpu_block)
                need_transfer_in_blocks[block_hash] = cpu_block

        return TransferInTask(blocks=need_transfer_in_blocks,
                              requests=need_transfer_in_requests,
                              transfer_manager=self)

    def transfer(self, task: TransferInTask):
        from wde.workflows.decoding.kv_cache.remote.client import \
            ZeroRemoteKVCacheClient
        client = ZeroRemoteKVCacheClient()

        blocks = task.blocks

        block_hashs = np.array(list(blocks.keys()), dtype=np.int64)

        response = client.get(self.server_name,
                              self.name,
                              block_hashs,
                              stream=True)

        metadata = next(response)

        count = 0
        for rep in response:
            block_hash = rep.block_hash[0]
            block = blocks[block_hash]
            data = rep.block
            self.cpu_cache[block.physical_block_id] = data
            count += 1
            task.hits.add(block_hash)

        assert metadata.hit == count
