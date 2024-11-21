import enum
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Set, Tuple

from wde.logger import init_logger
from wde.workflows.core.processor.input_processor import RequestProcessor
from wde.workflows.core.scheduler import Scheduler
from wde.workflows.core.schema.engine_io import RequestOutput
from wde.workflows.decoding.backends.core.block_manager import \
    SelfAttnBlockSpaceManager
from wde.workflows.decoding.backends.core.interfaces import AllocStatus
from wde.workflows.decoding.backends.sequence import (Sequence, SequenceData,
                                                      SequenceGroup,
                                                      SequenceGroupMetadata,
                                                      SequenceStatus)
from wde.workflows.decoding.config import CacheConfig, DecodingSchedulerConfig
from wde.workflows.decoding.schema.engine_io import (
    DecodingSchedulableRequest, DecodingSchedulerOutput)

logger = init_logger(__name__)


class PreemptionMode(enum.Enum):
    SWAP = enum.auto()
    RECOMPUTE = enum.auto()


@dataclass
class DecodingSchedulingBudget:
    token_budget: int
    max_num_requests: int
    _request_ids_num_batched_tokens: Set[str] = field(default_factory=set)
    _request_ids_num_curr_seqs: Set[str] = field(default_factory=set)
    _num_batched_tokens: int = 0
    _num_curr_seqs: int = 0

    def can_schedule(self, *, num_new_tokens: int, num_new_seqs: int):
        assert num_new_tokens != 0
        assert num_new_seqs != 0
        return (self.num_batched_tokens + num_new_tokens <= self.token_budget
                and self.num_curr_seqs + num_new_seqs <= self.max_num_requests)

    def remaining_token_budget(self):
        return self.token_budget - self.num_batched_tokens

    def add_num_batched_tokens(self, req_id: str, num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            return

        self._request_ids_num_batched_tokens.add(req_id)
        self._num_batched_tokens += num_batched_tokens

    def subtract_num_batched_tokens(self, req_id: str,
                                    num_batched_tokens: int):
        if req_id in self._request_ids_num_batched_tokens:
            self._request_ids_num_batched_tokens.remove(req_id)
            self._num_batched_tokens -= num_batched_tokens

    def add_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            return

        self._request_ids_num_curr_seqs.add(req_id)
        self._num_curr_seqs += num_curr_seqs

    def subtract_num_seqs(self, req_id: str, num_curr_seqs: int):
        if req_id in self._request_ids_num_curr_seqs:
            self._request_ids_num_curr_seqs.remove(req_id)
            self._num_curr_seqs -= num_curr_seqs

    @property
    def num_batched_tokens(self):
        return self._num_batched_tokens

    @property
    def num_curr_seqs(self):
        return self._num_curr_seqs


@dataclass
class SchedulerRunningOutputs:
    decode_requests: List[DecodingSchedulableRequest]
    prefill_requests: List[DecodingSchedulableRequest]
    preempted: List[DecodingSchedulableRequest]
    swapped_out: List[DecodingSchedulableRequest]
    blocks_to_swap_out: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]

    @classmethod
    def create_empty(cls) -> "SchedulerRunningOutputs":
        return SchedulerRunningOutputs(
            decode_requests=[],
            prefill_requests=[],
            preempted=[],
            swapped_out=[],
            blocks_to_swap_out=[],
            blocks_to_copy=[],
        )


@dataclass
class SchedulerSwappedInOutputs:
    decode_requests: List[DecodingSchedulableRequest]
    prefill_requests: List[DecodingSchedulableRequest]
    blocks_to_swap_in: List[Tuple[int, int]]
    blocks_to_copy: List[Tuple[int, int]]
    infeasible_requests: List[DecodingSchedulableRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerSwappedInOutputs":
        return SchedulerSwappedInOutputs(
            decode_requests=[],
            prefill_requests=[],
            blocks_to_swap_in=[],
            blocks_to_copy=[],
            infeasible_requests=[],
        )


@dataclass
class SchedulerPrefillOutputs:
    scheduled_requests: List[DecodingSchedulableRequest]
    ignored_requests: List[DecodingSchedulableRequest]

    @classmethod
    def create_empty(cls) -> "SchedulerPrefillOutputs":
        return SchedulerPrefillOutputs(
            scheduled_requests=[],
            ignored_requests=[],
        )


class DecodingScheduler(Scheduler):
    support_scheduling = ["sync_scheduling", "async_scheduling"]

    def __init__(
        self,
        scheduler_config: DecodingSchedulerConfig,
        cache_config: CacheConfig,
        request_processor: RequestProcessor,
    ) -> None:
        super().__init__(scheduler_config, request_processor)

        self.cache_config = cache_config

        BlockSpaceManagerImpl = SelfAttnBlockSpaceManager

        num_gpu_blocks = cache_config.num_gpu_blocks
        num_cpu_blocks = cache_config.num_cpu_blocks

        self.block_manager = BlockSpaceManagerImpl(
            block_size=self.cache_config.block_size,
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            sliding_window=self.cache_config.sliding_window,
            enable_caching=self.cache_config.enable_prefix_caching)

        self.running: Deque[DecodingSchedulableRequest] = deque()
        self.swapped: Deque[DecodingSchedulableRequest] = deque()
        self.preemption_mode = scheduler_config.preemption_mode
        self.num_cumulative_preemption: int = 0

    @classmethod
    def from_engine(cls, engine):
        return cls(engine.engine_config.scheduler_config,
                   engine.engine_config.cache_config, engine.request_processor)

    def _schedule_swapped(
        self,
        budget: DecodingSchedulingBudget,
        enable_chunking: bool = False,
    ) -> SchedulerSwappedInOutputs:

        blocks_to_swap_in: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []
        decode_requests: List[DecodingSchedulableRequest] = []
        prefill_requests: List[DecodingSchedulableRequest] = []
        infeasible_requests: List[DecodingSchedulableRequest] = []

        swapped_queue = self.swapped

        while swapped_queue:
            request = swapped_queue[0]
            scheduled_ts = time.perf_counter()

            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                swapped_queue.popleft()
                continue

            seq_group = request.seq_group

            alloc_status = self.block_manager.can_swap_in(seq_group, 0)
            if alloc_status == AllocStatus.LATER:
                break
            elif alloc_status == AllocStatus.NEVER:
                logger.warning(
                    "Failing the request %s because there's not enough kv "
                    "cache blocks to run the entire sequence.",
                    seq_group.request_id)
                for seq in seq_group.get_seqs():
                    seq.status = SequenceStatus.FINISHED_IGNORED
                infeasible_requests.append(request)
                swapped_queue.popleft()
                continue

            # The total number of sequences in the RUNNING state should not
            # exceed the maximum number of sequences.
            num_new_seqs = seq_group.get_max_num_running_seqs()
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.SWAPPED,
                                                      enable_chunking, budget)

            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            request.set_scheduled_ts(scheduled_ts)
            swapped_queue.popleft()

            self._swap_in(seq_group, blocks_to_swap_in)
            self._append_slots(seq_group, blocks_to_copy)
            is_prefill = seq_group.is_prefill()

            if is_prefill:
                request.token_chunk_size = num_new_tokens
                prefill_requests.append(request)
            else:
                request.token_chunk_size = 1
                decode_requests.append(request)

            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        return SchedulerSwappedInOutputs(
            decode_requests=decode_requests,
            prefill_requests=prefill_requests,
            blocks_to_swap_in=blocks_to_swap_in,
            blocks_to_copy=blocks_to_copy,
            infeasible_requests=infeasible_requests,
        )

    def _schedule_prefills(
        self,
        budget: DecodingSchedulingBudget,
        enable_chunking: bool = False,
    ) -> SchedulerPrefillOutputs:
        ignored_requests: List[DecodingSchedulableRequest] = []
        scheduled_requests: List[DecodingSchedulableRequest] = []

        waiting_queue = self.waiting
        while waiting_queue:
            request = waiting_queue[0]
            scheduled_ts = time.perf_counter()

            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                waiting_queue.popleft()
                continue

            if not isinstance(request, DecodingSchedulableRequest):
                request = self.request_processor(request)
                waiting_queue[0] = request

            seq_group = request.seq_group

            waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)
            assert len(waiting_seqs) == 1, (
                "Waiting sequence group should have only one prompt "
                "sequence.")
            num_new_tokens = self._get_num_new_tokens(seq_group,
                                                      SequenceStatus.WAITING,
                                                      enable_chunking, budget)
            if not enable_chunking:
                num_prompt_tokens = waiting_seqs[0].get_len()
                assert num_new_tokens == num_prompt_tokens

            prompt_limit = self.scheduler_config.max_model_len
            if num_new_tokens > prompt_limit:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds limit of %d", num_new_tokens, prompt_limit)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_requests.append(request)
                waiting_queue.popleft()
                continue

            # If the sequence group cannot be allocated, stop.
            can_allocate = self.block_manager.can_allocate(seq_group)
            if can_allocate == AllocStatus.LATER:
                break
            elif can_allocate == AllocStatus.NEVER:
                logger.warning(
                    "Input prompt (%d tokens) is too long"
                    " and exceeds the capacity of block_manager",
                    num_new_tokens)
                for seq in waiting_seqs:
                    seq.status = SequenceStatus.FINISHED_IGNORED
                ignored_requests.append(request)
                waiting_queue.popleft()
                continue

            num_new_seqs = seq_group.get_max_num_running_seqs()
            if (num_new_tokens == 0
                    or not budget.can_schedule(num_new_tokens=num_new_tokens,
                                               num_new_seqs=num_new_seqs)):
                break

            # Can schedule this request.
            waiting_queue.popleft()
            self._allocate_and_set_running(seq_group)

            request.set_scheduled_ts(scheduled_ts)

            request.token_chunk_size = num_new_tokens
            scheduled_requests.append(request)
            budget.add_num_batched_tokens(seq_group.request_id, num_new_tokens)
            budget.add_num_seqs(seq_group.request_id, num_new_seqs)

        return SchedulerPrefillOutputs(scheduled_requests=scheduled_requests,
                                       ignored_requests=ignored_requests)

    def _schedule_running(self,
                          budget: DecodingSchedulingBudget,
                          enable_chunking=True) -> SchedulerRunningOutputs:
        busy_requests = []
        running_queue = []
        for request in self.running:
            if request.request_id in self.aborted_requests:
                self.actual_abort_request(request.request_id)
                continue

            if request.request_id not in self.requests:
                # aborted_requests
                continue

            if request.busy:
                busy_requests.append(request)
            else:
                running_queue.append(request)

        if not running_queue:
            return SchedulerRunningOutputs.create_empty()

        running_queue = deque(
            sorted(running_queue,
                   key=lambda request: request.metrics.arrival_ts))

        blocks_to_swap_out: List[Tuple[int, int]] = []
        blocks_to_copy: List[Tuple[int, int]] = []

        decode_requests = []
        prefill_requests = []
        preempted = []
        swapped_out = []

        while running_queue:
            request = running_queue[0]
            scheduled_ts = time.perf_counter()

            seq_group = request.seq_group

            num_running_tokens = self._get_num_new_tokens(
                seq_group, SequenceStatus.RUNNING, enable_chunking, budget)

            if num_running_tokens == 0:
                # No budget => Stop
                break

            running_queue.popleft()
            request.set_scheduled_ts(scheduled_ts)

            while not self._can_append_slots(seq_group):
                budget.subtract_num_batched_tokens(seq_group.request_id,
                                                   num_running_tokens)
                num_running_seqs = seq_group.get_max_num_running_seqs()
                budget.subtract_num_seqs(seq_group.request_id,
                                         num_running_seqs)

                if running_queue:
                    # Preempt the lowest-priority sequence groups.
                    victim_request = running_queue.pop()
                    preempted_mode = self._preempt(victim_request.seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(victim_request)
                    else:
                        swapped_out.append(victim_request)
                else:
                    # No other sequence groups can be preempted.
                    # Preempt the current sequence group.
                    preempted_mode = self._preempt(seq_group,
                                                   blocks_to_swap_out)
                    if preempted_mode == PreemptionMode.RECOMPUTE:
                        preempted.append(request)
                    else:
                        swapped_out.append(request)
                    break
            else:
                self._append_slots(seq_group, blocks_to_copy)
                is_prefill = seq_group.is_prefill()
                if is_prefill:
                    request.token_chunk_size = num_running_tokens
                    prefill_requests.append(request)
                else:
                    request.token_chunk_size = 1
                    decode_requests.append(request)

                budget.add_num_batched_tokens(seq_group.request_id,
                                              num_running_tokens)
                # OPTIMIZATION:  Note that get_max_num_running_seqs is
                # expensive. For the default scheduling chase where
                # enable_chunking is False, num_seqs are updated before running
                # this method, so we don't have to update it again here.
                if enable_chunking:
                    num_running_seqs = seq_group.get_max_num_running_seqs()
                    budget.add_num_seqs(seq_group.request_id, num_running_seqs)

        self.running = running_queue
        self.running.extend(busy_requests)

        return SchedulerRunningOutputs(decode_requests=decode_requests,
                                       prefill_requests=prefill_requests,
                                       preempted=preempted,
                                       swapped_out=swapped_out,
                                       blocks_to_swap_out=blocks_to_swap_out,
                                       blocks_to_copy=blocks_to_copy)

    def _schedule(self) -> DecodingSchedulerOutput:
        budget = DecodingSchedulingBudget(
            token_budget=self.scheduler_config.max_num_batched_tokens,
            max_num_requests=self.scheduler_config.max_num_requests,
        )

        running_scheduled = self._schedule_running(budget)

        if len(running_scheduled.preempted) + len(
                running_scheduled.swapped_out) == 0:
            swapped_in = self._schedule_swapped(budget)
        else:
            swapped_in = SchedulerSwappedInOutputs.create_empty()

        prefills = self._schedule_prefills(budget, enable_chunking=True)

        assert (budget.num_batched_tokens
                <= self.scheduler_config.max_num_batched_tokens)
        assert budget.num_curr_seqs <= self.scheduler_config.max_num_requests

        # Update waiting requests.
        self.waiting.extendleft(running_scheduled.preempted)

        # Update new running requests.
        self.running.extend(swapped_in.decode_requests)
        self.running.extend(swapped_in.prefill_requests)
        self.running.extend(running_scheduled.decode_requests)
        self.running.extend(running_scheduled.prefill_requests)

        self.running.extend(prefills.scheduled_requests)

        # Update swapped requests.
        self.swapped.extend(running_scheduled.swapped_out)

        return DecodingSchedulerOutput(
            scheduled_requests=(prefills.scheduled_requests +
                                running_scheduled.prefill_requests +
                                swapped_in.prefill_requests +
                                running_scheduled.decode_requests +
                                swapped_in.decode_requests),
            num_prefill_groups=(len(prefills.scheduled_requests) +
                                len(swapped_in.prefill_requests) +
                                len(running_scheduled.prefill_requests)),
            num_batched_tokens=budget.num_batched_tokens,
            blocks_to_swap_in=swapped_in.blocks_to_swap_in,
            blocks_to_swap_out=running_scheduled.blocks_to_swap_out,
            blocks_to_copy=running_scheduled.blocks_to_copy +
            swapped_in.blocks_to_copy,
            ignored_requests=prefills.ignored_requests +
            swapped_in.infeasible_requests,
            running_queue_size=len(self.running),
            preempted=(len(running_scheduled.preempted) +
                       len(running_scheduled.swapped_out)),
        )

    @property
    def num_decoding_tokens_per_seq(self) -> int:
        """The number of new tokens."""
        return 1

    def _can_append_slots(self, seq_group: SequenceGroup) -> bool:
        return self.block_manager.can_append_slots(
            seq_group=seq_group,
            num_lookahead_slots=0,
        )

    def schedule(self) -> Optional[DecodingSchedulerOutput]:
        scheduling_begin_ts = time.perf_counter()

        scheduler_outputs = self._schedule()
        now = time.time()

        seq_group_metadata_list: List[SequenceGroupMetadata] = []
        for i, request in enumerate(scheduler_outputs.scheduled_requests):
            request.busy = True

            seq_group = request.seq_group
            token_chunk_size = request.token_chunk_size

            seq_data: Dict[int, SequenceData] = {}
            block_tables: Dict[int, List[int]] = {}

            for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
                seq_id = seq.seq_id
                seq_data[seq_id] = seq.data
                block_tables[seq_id] = self.block_manager.get_block_table(seq)
                self.block_manager.access_all_blocks_in_seq(seq, now)

            common_computed_block_nums = (
                self.block_manager.get_common_computed_block_ids(
                    seq_group.get_seqs(status=SequenceStatus.RUNNING)))

            do_sample = True
            if seq_group.is_prefill():
                seqs = seq_group.get_seqs()

                assert len(seqs) == 1
                if (token_chunk_size + seqs[0].data.get_num_computed_tokens()
                        < seqs[0].data.get_len()):
                    do_sample = False

            is_prompt = seq_group.is_prefill()
            seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group.request_id,
                is_prompt=is_prompt,
                seq_data=seq_data,
                sampling_params=seq_group.sampling_params,
                block_tables=block_tables,
                do_sample=do_sample,
                token_chunk_size=token_chunk_size,
                computed_block_nums=common_computed_block_nums,
            )
            seq_group_metadata_list.append(seq_group_metadata)

        for scheduled_seq_group in scheduler_outputs.scheduled_requests:
            self.block_manager.mark_blocks_as_computed(
                scheduled_seq_group.seq_group,
                scheduled_seq_group.token_chunk_size)

        scheduler_outputs.seq_group_metadata_list = seq_group_metadata_list

        scheduling_end_ts = time.perf_counter()
        scheduling_time = scheduling_end_ts - scheduling_begin_ts
        num_requests = len(scheduler_outputs.scheduled_requests)
        num_batched_tokens = scheduler_outputs.num_batched_tokens
        for request in scheduler_outputs.scheduled_requests:
            request.metrics.scheduling_time = scheduling_time
            request.metrics.num_requests = num_requests
            request.metrics.num_batched_tokens = num_batched_tokens

        return scheduler_outputs

    def fork_seq(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        self.block_manager.fork(parent_seq, child_seq)

    def free_seq(self, seq: Sequence) -> None:
        self.block_manager.free(seq)

    def free_finished_request(self,
                              request_outputs: List[RequestOutput]) -> None:

        request_ids = set(request.request_id for request in request_outputs)

        remaining: Deque[DecodingSchedulableRequest] = deque()
        for request in self.running:
            seq_group = request.seq_group
            if not seq_group.is_finished():
                remaining.append(request)
                if request.request_id in request_ids:
                    request.busy = False
            else:
                self.requests.remove(request.request_id)

        self.running = remaining

    def _allocate_and_set_running(self, seq_group: SequenceGroup) -> None:
        self.block_manager.allocate(seq_group)
        for seq in seq_group.get_seqs(status=SequenceStatus.WAITING):
            seq.status = SequenceStatus.RUNNING

    def _append_slots(
        self,
        seq_group: SequenceGroup,
        blocks_to_copy: List[Tuple[int, int]],
    ) -> None:
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            cows = self.block_manager.append_slots(seq, 0)
            blocks_to_copy.extend(cows)

    def _preempt(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> PreemptionMode:
        if self.preemption_mode is None:
            if seq_group.get_max_num_running_seqs() == 1:
                preemption_mode = PreemptionMode.RECOMPUTE
            else:
                preemption_mode = PreemptionMode.SWAP

        elif self.preemption_mode == "swap":
            preemption_mode = PreemptionMode.SWAP
        else:
            preemption_mode = PreemptionMode.RECOMPUTE

        if self.num_cumulative_preemption % 50 == 0:
            logger.warning(
                "Sequence group %s is preempted by %s mode because there is "
                "not enough KV cache space. This can affect the end-to-end "
                "performance. Increase gpu_memory_utilization or "
                "tensor_parallel_size to provide more KV cache memory. "
                "total_num_cumulative_preemption=%d", seq_group.request_id,
                preemption_mode, self.num_cumulative_preemption + 1)
        self.num_cumulative_preemption += 1

        if preemption_mode == PreemptionMode.RECOMPUTE:
            self._preempt_by_recompute(seq_group)
        elif preemption_mode == PreemptionMode.SWAP:
            self._preempt_by_swap(seq_group, blocks_to_swap_out)
        else:
            raise AssertionError("Invalid preemption mode.")
        return preemption_mode

    def _preempt_by_recompute(
        self,
        seq_group: SequenceGroup,
    ) -> None:
        seqs = seq_group.get_seqs(status=SequenceStatus.RUNNING)
        assert len(seqs) == 1
        for seq in seqs:
            seq.status = SequenceStatus.WAITING
            self.free_seq(seq)
            seq.reset_state_for_recompute()

    def _preempt_by_swap(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        self._swap_out(seq_group, blocks_to_swap_out)

    def _swap_in(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_in: List[Tuple[int, int]],
    ) -> None:
        mapping = self.block_manager.swap_in(seq_group)
        blocks_to_swap_in.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.SWAPPED):
            seq.status = SequenceStatus.RUNNING

    def _swap_out(
        self,
        seq_group: SequenceGroup,
        blocks_to_swap_out: List[Tuple[int, int]],
    ) -> None:
        if not self.block_manager.can_swap_out(seq_group):
            raise RuntimeError(
                "Aborted due to the lack of CPU swap space. Please increase "
                "the swap space to avoid this error.")
        mapping = self.block_manager.swap_out(seq_group)
        blocks_to_swap_out.extend(mapping)
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            seq.status = SequenceStatus.SWAPPED

    def _get_num_new_tokens(self, seq_group: SequenceGroup,
                            status: SequenceStatus, enable_chunking: bool,
                            budget: DecodingSchedulingBudget) -> int:
        num_new_tokens = 0
        seqs = seq_group.get_seqs(status=status)
        for seq in seqs:
            num_new_tokens += seq.get_num_new_tokens()
        assert num_new_tokens > 0
        # Chunk if a running request cannot fit in.
        # If number of seq > 1, it means it is doing beam search in a
        # decode phase. Do not chunk in that case.
        if enable_chunking and len(seqs) == 1:
            num_new_tokens = min(num_new_tokens,
                                 budget.remaining_token_budget())
        return num_new_tokens
