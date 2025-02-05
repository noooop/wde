import time
from typing import List, Optional

from wde.logger import init_logger
from wde.workflows.decoding.kv_cache.naive.scheduler import (
    DecodingSchedulingBudget, NaiveDecodingScheduler, SchedulerRunningOutputs)
from wde.workflows.decoding.schema.engine_io import (
    DecodingSchedulableRequest, DecodingSchedulerOutput)

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

            # 2. Check if hit prefix caching ready
            if not request.vblock.ready():
                running_queue.popleft()
                not_ready_requests.append(request)
                continue

            is_prefill = request.get_is_prefill()

            # 3. test kv_cache is above high_watermark
            if is_prefill and self.kv_cache_manager.high_watermark():
                break

            num_new_tokens = request.num_new_tokens
            assert num_new_tokens > 0

            # 4. chunked prefill
            budget_bound_token_chunk_size = min(
                num_new_tokens, budget.remaining_token_budget())

            if budget_bound_token_chunk_size == 0:
                # No budget => Stop
                break

            running_queue.popleft()

            # 5. try allocate
            while not self._can_allocate(request,
                                         budget_bound_token_chunk_size):
                if running_queue:
                    # Preempt the lowest-priority request.
                    victim_request = running_queue.pop()
                    preempted.append(victim_request)

                    while victim_request.num_computed_tokens > 0:
                        self.kv_cache_manager.free_last_block(victim_request)

                        if self._can_allocate(request,
                                              budget_bound_token_chunk_size):
                            break
                else:
                    preempted.append(request)
                    break
            else:
                # 6. Can schedule this request.
                request.token_chunk_size = budget_bound_token_chunk_size

                self.kv_cache_manager.allocate(request)

                request.vblock.acquire()

                if is_prefill:
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

    def schedule(self) -> Optional[DecodingSchedulerOutput]:
        scheduler_outputs = super().schedule()

        # e.g. Waiting for copy on write done in yoco
        self.kv_cache_manager.join()
        return scheduler_outputs