import queue
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Optional

from wde.workflows.decoding.kv_cache.remote.transfer import (
    TransferOutManager, TransferTask)

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.offloading.swap import SwapTask


class RemoteManager:

    def __init__(self, engine_config, cpu_cache, cpu_block_allocator):
        self.engine_config = engine_config
        self.cpu_block_allocator = cpu_block_allocator

        self.transfer_out_manager = TransferOutManager(
            name=engine_config.model_config.model,
            server_name=engine_config.cache_config.remote_kv_cache_server_name,
            cpu_cache=cpu_cache,
            cpu_block_allocator=cpu_block_allocator,
            remote_manager=self,
        )

        self.threads = ThreadPoolExecutor(
            max_workers=self.engine_config.sys_config.
            kv_cache_transfer_max_workers)
        self.finishd_task = queue.Queue()

    def add_transfer_callback(self, swap_out_task: Optional["SwapTask"]):
        if swap_out_task is None:
            return

        return TransferTask.from_swap_out_task(
            swap_out_task=swap_out_task,
            transfer_manager=self.transfer_out_manager)

    def check_finishd_task(self):
        while True:
            try:
                task = self.finishd_task.get(block=False)
            except queue.Empty:
                break
            task.do_callback()

    def join(self):
        self.threads.shutdown()
