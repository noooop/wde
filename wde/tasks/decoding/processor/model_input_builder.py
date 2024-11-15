import dataclasses
import weakref
from typing import Dict, List, Optional

import torch
from vllm.utils import flatten_2d_lists, is_pin_memory_available

from wde.tasks.core.config import DeviceConfig
from wde.tasks.core.processor.model_input_builder import ModelInputBuilder
from wde.tasks.decoding.backends.sampling_metadata import SamplingMetadata
from wde.tasks.decoding.backends.sequence import SequenceGroupMetadata
from wde.tasks.decoding.config import (CacheConfig, DecodingModelConfig,
                                       DecodingSchedulerConfig)
from wde.tasks.decoding.scheduler import DecodingSchedulerOutput
from wde.tasks.decoding.schema.execute_io import (
    DecodingModelInputForGPU, DecodingModelInputForGPUWithSamplingMetadata,
    DecodingWorkerInputForGPU, ExecuteInput)

pin_memory = is_pin_memory_available()


class DecodingModelPreProcessor(ModelInputBuilder):

    def __init__(
        self,
        device_config: DeviceConfig,
        model_config: DecodingModelConfig,
        scheduler_config: DecodingSchedulerConfig,
        cache_config: CacheConfig,
        attn_backend,
    ):
        self.device = device_config.device

        self.model_config = model_config
        self.scheduler_config = scheduler_config
        self.cache_config = cache_config
        self.attn_backend = attn_backend

    @classmethod
    def from_engine(cls, engine):
        return cls(
            engine.engine_config.device_config,
            engine.engine_config.model_config,
            engine.engine_config.scheduler_config,
            engine.engine_config.cache_config,
            attn_backend=engine.executor.worker.model_runner.attn_backend)

    def _prepare_model_input_tensors(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> DecodingModelInputForGPUWithSamplingMetadata:
        builder = DecodingModelInputForGPUBuilder(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            cache_config=self.cache_config,
            attn_backend=self.attn_backend,
            device=self.device)
        for seq_group_metadata in seq_group_metadata_list:
            builder.add_seq_group(seq_group_metadata)
        return builder.build()  # type: ignore

    def prepare_model_input(
        self, seq_group_metadata_list: List[SequenceGroupMetadata]
    ) -> DecodingModelInputForGPUWithSamplingMetadata:
        model_input = self._prepare_model_input_tensors(
            seq_group_metadata_list)

        sampling_metadata = SamplingMetadata.prepare(seq_group_metadata_list,
                                                     model_input.seq_lens,
                                                     model_input.query_lens,
                                                     pin_memory)

        is_prompt = (seq_group_metadata_list[0].is_prompt
                     if seq_group_metadata_list else None)
        return dataclasses.replace(model_input,
                                   sampling_metadata=sampling_metadata,
                                   is_prompt=is_prompt)

    @torch.inference_mode()
    def prepare_worker_input(
        self, scheduler_output: DecodingSchedulerOutput
    ) -> DecodingWorkerInputForGPU:
        num_requests = len(scheduler_output.seq_group_metadata_list)

        blocks_to_swap_in = torch.tensor(scheduler_output.blocks_to_swap_in,
                                         device="cpu",
                                         dtype=torch.int64,
                                         pin_memory=pin_memory).view(-1, 2)
        blocks_to_swap_out = torch.tensor(scheduler_output.blocks_to_swap_out,
                                          device="cpu",
                                          dtype=torch.int64,
                                          pin_memory=pin_memory).view(-1, 2)

        blocks_to_copy = torch.tensor(scheduler_output.blocks_to_copy,
                                      device="cpu",
                                      dtype=torch.int64,
                                      pin_memory=pin_memory).view(-1, 2)

        return DecodingWorkerInputForGPU(num_requests=num_requests,
                                         blocks_to_swap_in=blocks_to_swap_in,
                                         blocks_to_swap_out=blocks_to_swap_out,
                                         blocks_to_copy=blocks_to_copy)

    def __call__(self,
                 scheduler_output: DecodingSchedulerOutput) -> ExecuteInput:
        worker_input = self.prepare_worker_input(scheduler_output)
        model_input = self.prepare_model_input(
            scheduler_output.seq_group_metadata_list)

        return ExecuteInput(worker_input=worker_input, model_input=model_input)


class DecodingModelInputForGPUBuilder:

    class InterDataForSeqGroup:

        def __init__(
            self,
            *,
            # From sequence group metadata.
            request_id: str,
            seq_ids: List[int],
            is_prompt: bool,
            block_tables: Optional[Dict[int, List[int]]],
            computed_block_nums: List[int],
            n_seqs: int = 0,

            # Input tokens and positions.
            input_tokens: Optional[List[List[int]]] = None,
            input_positions: Optional[List[List[int]]] = None,

            # The sequence length (may be capped to the sliding window).
            seq_lens: Optional[List[int]] = None,
            # The original sequence length (before applying sliding window).
            # This is used to compute slot mapping.
            orig_seq_lens: Optional[List[int]] = None,
            # The query length.
            query_lens: Optional[List[int]] = None,
            # The number of tokens that are already computed.
            context_lens: Optional[List[int]] = None,
            # The current sliding window block.
            curr_sliding_window_blocks: Optional[List[int]] = None,

            # Whether the prefix cache is hit (prefill only).
            prefix_cache_hit: bool = False,
        ):
            self.request_id = request_id
            self.seq_ids = seq_ids
            self.is_prompt = is_prompt
            self.block_tables = block_tables
            self.computed_block_nums = computed_block_nums
            self.n_seqs = n_seqs
            self.input_tokens = input_tokens or []
            self.input_positions = input_positions or []
            self.seq_lens = seq_lens or []
            self.orig_seq_lens = orig_seq_lens or []
            self.query_lens = query_lens or []
            self.context_lens = context_lens or []
            self.curr_sliding_window_blocks = curr_sliding_window_blocks or []

            self.prefix_cache_hit = prefix_cache_hit

            self.__post_init__()

        def __post_init__(self):
            self.n_seqs = len(self.seq_ids)

            self.input_tokens = [[] for _ in range(self.n_seqs)]
            self.input_positions = [[] for _ in range(self.n_seqs)]
            self.seq_lens = [0] * self.n_seqs
            self.orig_seq_lens = [0] * self.n_seqs
            self.query_lens = [0] * self.n_seqs
            self.context_lens = [0] * self.n_seqs
            self.curr_sliding_window_blocks = [0] * self.n_seqs

    def __init__(self, model_config: DecodingModelConfig,
                 scheduler_config: DecodingSchedulerConfig,
                 cache_config: CacheConfig, attn_backend, device):
        # Compute functions for each sequence in a sequence group.
        # WARNING: The order of the functions matters!
        self.per_seq_compute_fns = [
            self._compute_lens,
            self._compute_for_prefix_cache_hit,
            self._compute_for_sliding_window,
        ]

        self.scheduler_config = scheduler_config
        self.sliding_window = model_config.get_sliding_window()
        self.block_size = cache_config.block_size
        self.device = device

        self.decode_only = True

        # Intermediate data (data in CPU before going to GPU) for
        # the current sequence group.
        self.inter_data_list: List[
            DecodingModelInputForGPUBuilder.InterDataForSeqGroup] = []

        # Attention metadata inputs.
        self.attn_metadata_builder = attn_backend.make_metadata_builder(
            weakref.proxy(self))

        # Engine/Model configurations.
        self.chunked_prefill_enabled = True

        if self.sliding_window is not None:
            self.sliding_window_blocks = (
                self.sliding_window + self.block_size - 1) // self.block_size
            self.block_aligned_sliding_window = \
                self.sliding_window_blocks * self.block_size

    def _compute_lens(self, inter_data: InterDataForSeqGroup, seq_idx: int,
                      seq_group_metadata: SequenceGroupMetadata):
        """Compute context length, sequence length and tokens
        for the given sequence data.
        """
        seq_data = seq_group_metadata.seq_data[inter_data.seq_ids[seq_idx]]
        token_chunk_size = seq_group_metadata.token_chunk_size

        # Compute context length (the number of tokens that are
        # already computed) and sequence length (total number of tokens).
        seq_len = seq_data.get_len()
        if inter_data.is_prompt:
            context_len = seq_data.get_num_computed_tokens()
        else:
            # get_num_computed_tokens is incorrect for spec decoding.
            # So, we should have a special logic here.
            # TODO(sang): Fix it.
            context_len = seq_len - 1
        seq_len = min(seq_len, context_len + token_chunk_size)

        # Compute tokens.
        if inter_data.is_prompt:
            tokens = seq_data.get_token_ids()[context_len:seq_len]
        else:
            # Optimization. get_token_ids requires the entire copy of
            # tokens.
            tokens = [seq_data.get_last_token_id()]

        inter_data.seq_lens[seq_idx] = seq_len
        inter_data.orig_seq_lens[seq_idx] = seq_len
        inter_data.context_lens[seq_idx] = context_len
        inter_data.input_tokens[seq_idx] = tokens
        inter_data.input_positions[seq_idx] = list(range(context_len, seq_len))
        inter_data.query_lens[
            seq_idx] = seq_len - context_len if inter_data.is_prompt else 1

    def _compute_for_prefix_cache_hit(
            self, inter_data: InterDataForSeqGroup, seq_idx: int,
            seq_group_metadata: SequenceGroupMetadata):
        """Check if hit prefix cache (i.e., some blocks are already computed).
        If hit, update input tokens and positions to only compute the
        remaining blocks.
        """
        computed_block_nums = inter_data.computed_block_nums

        # Note that prefix caching does not support sliding window.
        prefix_cache_hit = (computed_block_nums is not None
                            and len(computed_block_nums) > 0
                            and self.sliding_window is None
                            and inter_data.is_prompt)
        inter_data.prefix_cache_hit = prefix_cache_hit
        if self.chunked_prefill_enabled and prefix_cache_hit:
            raise RuntimeError(
                "chunked prefill cannot be used with prefix caching now.")

        # If prefix cache is hit, advance context length to bypass
        # hit blocks. Accordingly, input tokens, position and query length
        # have to be updated.
        if prefix_cache_hit:
            assert computed_block_nums is not None
            context_len = len(computed_block_nums) * self.block_size
            inter_data.input_tokens[seq_idx] = inter_data.input_tokens[
                seq_idx][context_len:]
            inter_data.input_positions[seq_idx] = inter_data.input_positions[
                seq_idx][context_len:]
            inter_data.context_lens[seq_idx] = context_len
            inter_data.query_lens[
                seq_idx] = inter_data.seq_lens[seq_idx] - context_len

    def _compute_for_sliding_window(self, inter_data: InterDataForSeqGroup,
                                    seq_idx: int,
                                    seq_group_metadata: SequenceGroupMetadata):
        """Update seq_len and curr_sliding_window_block for the given
        sequence data (only required by decoding) if sliding window is enabled.
        """
        curr_sliding_window_block = 0
        sliding_seq_len = inter_data.seq_lens[seq_idx]
        if not inter_data.is_prompt and self.sliding_window is not None:
            # TODO(sang): This is a hack to make sliding window work with
            # paged attn. We can remove it if we make paged attn kernel
            # to properly handle slinding window attn.
            curr_sliding_window_block = self.sliding_window_blocks
            suff_len = inter_data.seq_lens[seq_idx] % self.block_size
            sliding_seq_len = min(inter_data.seq_lens[seq_idx],
                                  self.block_aligned_sliding_window + suff_len)
            if suff_len > 0:
                curr_sliding_window_block += 1

        inter_data.curr_sliding_window_blocks[
            seq_idx] = curr_sliding_window_block
        inter_data.seq_lens[seq_idx] = sliding_seq_len

    def add_seq_group(self, seq_group_metadata: SequenceGroupMetadata):
        """Add a sequence group to the builder."""
        seq_ids = list(seq_group_metadata.seq_data.keys())
        n_seqs = len(seq_ids)
        is_prompt = seq_group_metadata.is_prompt

        if is_prompt:
            assert n_seqs == 1
            self.decode_only = False

        inter_data = self.InterDataForSeqGroup(
            request_id=seq_group_metadata.request_id,
            seq_ids=seq_ids,
            is_prompt=is_prompt,
            block_tables=seq_group_metadata.block_tables,
            computed_block_nums=seq_group_metadata.computed_block_nums)
        self.inter_data_list.append(inter_data)

        for seq_idx in range(n_seqs):
            for per_seq_fn in self.per_seq_compute_fns:
                per_seq_fn(inter_data, seq_idx, seq_group_metadata)

    def build(self) -> DecodingModelInputForGPU:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = flatten_2d_lists([
            flatten_2d_lists(inter_data.input_tokens)
            for inter_data in self.inter_data_list
        ])
        if not input_tokens:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return DecodingModelInputForGPUWithSamplingMetadata()
        input_positions = flatten_2d_lists([
            flatten_2d_lists(inter_data.input_positions)
            for inter_data in self.inter_data_list
        ])
        seq_lens = []
        max_decode_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
        query_lens = flatten_2d_lists(
            [inter_data.query_lens for inter_data in self.inter_data_list])

        batch_size = len(input_tokens)
        input_tokens_tensor = torch.tensor(input_tokens,
                                           dtype=torch.long,
                                           device="cpu")
        input_positions_tensor = torch.tensor(input_positions,
                                              dtype=torch.long,
                                              device="cpu")

        cuda_graph_pad_size = -1

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, cuda_graph_pad_size, batch_size)

        return DecodingModelInputForGPUWithSamplingMetadata(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens)
