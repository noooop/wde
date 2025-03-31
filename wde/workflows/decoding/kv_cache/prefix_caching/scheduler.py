import time
from typing import List

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.naive.scheduler import (
    DecodingSchedulingBudget, NaiveDecodingScheduler, SchedulerRunningOutputs)
from wde.workflows.decoding.schema.engine_io import DecodingSchedulableRequest

logger = init_logger(__name__)


class PrefixCachingDecodingScheduler(NaiveDecodingScheduler):
    name = "Prefix Caching"
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def _schedule_running(self, budget: DecodingSchedulingBudget,
                          running_queue) -> SchedulerRunningOutputs:
        prefill_requests = []
        decode_requests = []
        preempted = []
        not_ready_requests: List[DecodingSchedulableRequest] = []

        while running_queue:
            if budget.full():
                break

            request = running_queue[0]
            if self.record_metrics:
                scheduled_ts = time.perf_counter()

            # 1. Write new token ids to vblock & try to hit prefix caching
            self.kv_cache_manager.update(request)

            # 1.1. Check if hit prefix caching ready
            if not request.vblock.ready():
                running_queue.popleft()
                not_ready_requests.append(request)
                continue

            # 2. test kv_cache is above high_watermark
            is_prefill = request.get_is_prefill()
            if is_prefill and self.kv_cache_manager.high_watermark():
                break

            # 3. chunked prefill budget
            token_budget = budget.remaining_token_budget()
            assert token_budget > 0

            running_queue.popleft()

            num_new_tokens = request.num_new_tokens
            assert num_new_tokens > 0

            token_chunk_size = 0

            if not request.has_unscheduled_tokens():
                # All tokens are hit kv_cache
                token_chunk_size = 1
            else:
                # 4. try to allocate
                while request.has_unscheduled_tokens() and token_budget > 0:
                    # allocate one block at a time

                    if self.kv_cache_manager.num_free_blocks == 0:
                        # There is no empty kvcache, perform preemption
                        while running_queue:
                            victim_request = running_queue[-1]

                            while victim_request.num_computed_tokens > 0:
                                self.kv_cache_manager.free_last_block(
                                    victim_request)

                                if self.kv_cache_manager.num_free_blocks > 0:
                                    break

                            if victim_request.num_computed_tokens == 0:
                                running_queue.pop()
                                preempted.append(victim_request)

                            if self.kv_cache_manager.num_free_blocks > 0:
                                break

                    if self.kv_cache_manager.num_free_blocks == 0:
                        # There is no space after preemption
                        break

                    part_size = self.kv_cache_manager.allocate(
                        request, token_budget)
                    token_chunk_size += part_size
                    token_budget -= part_size

            if token_chunk_size == 0:
                preempted.append(request)
                break

            request.vblock.acquire()
            request.token_chunk_size = token_chunk_size

            if request.get_is_prefill():
                prefill_requests.append(request)
            else:
                decode_requests.append(request)

            if self.record_metrics:
                request.set_scheduled_ts(scheduled_ts)

            budget.add_num_batched_tokens(request.request_id,
                                          request.token_chunk_size)

            budget.add_num_requests(request.request_id, 1)

        # add not ready requests back to waiting queue
        for request in not_ready_requests:
            running_queue.append(request)

        return SchedulerRunningOutputs(decode_requests=decode_requests,
                                       prefill_requests=prefill_requests,
                                       preempted=preempted)