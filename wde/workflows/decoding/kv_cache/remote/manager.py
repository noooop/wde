import queue
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, List, Optional

from wde.workflows.decoding.kv_cache.remote.transfer import (
    TransferInManager, TransferOutManager, TransferOutTask)

if TYPE_CHECKING:
    from wde.workflows.decoding.kv_cache.offloading.swap import SwapTask
    from wde.workflows.decoding.schema.request import \
        DecodingSchedulableRequest


class RemoteManager:

    def __init__(self, engine_config, cpu_cache, cpu_block_allocator,
                 gpu_kv_cache_manager):
        self.engine_config = engine_config
        self.cpu_block_allocator = cpu_block_allocator
        self.gpu_kv_cache_manager = gpu_kv_cache_manager

        self.transfer_out_manager = TransferOutManager(
            name=engine_config.model_config.model,
            server_name=engine_config.cache_config.remote_kv_cache_server_name,
            cpu_cache=cpu_cache,
            cpu_block_allocator=cpu_block_allocator,
            remote_manager=self,
        )

        self.transfer_in_manager = TransferInManager(
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

        return TransferOutTask.from_swap_out_task(
            swap_out_task=swap_out_task,
            transfer_manager=self.transfer_out_manager)

    def get_transfer_in_blocks(self, request):
        return self.transfer_in_manager.get_transfer_in_blocks(request)

    def get_transfer_in_task(
            self, scheduled_requests: List["DecodingSchedulableRequest"]):

        transfer_in_task = self.transfer_in_manager.get_transfer_in_task(
            scheduled_requests)

        if transfer_in_task is not None:
            transfer_in_task.submit()

        return transfer_in_task

    def check_finishd_task(self):
        while True:
            try:
                task_or_exception = self.finishd_task.get(block=False)
            except queue.Empty:
                break

            if isinstance(task_or_exception, Exception):
                exception = task_or_exception
                raise exception

            task = task_or_exception
            task.do_callback()

    def join(self):
        self.threads.shutdown()
