"""Attention layer with FlashAttention."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from vllm.utils import is_pin_memory_available, make_tensor_with_pad

from wde.workflows.core.backends.ops.flash_attn_interface import (
    flash_attn_varlen_func, flash_attn_with_kvcache, get_flash_attn_version)
from wde.workflows.decoding.backends.attention.backends.abstract import (
    AttentionType, DecodeOnlyAttentionBackend, DecodeOnlyAttentionImpl,
    DecodeOnlyAttentionMetadata, DecodeOnlyAttentionMetadataBuilder)
from wde.workflows.decoding.backends.attention.backends.utils import \
    compute_slot_mapping
from wde.workflows.decoding.schema.engine_io import DecodingSchedulableRequest

pin_memory = is_pin_memory_available()


class DecodeOnlyFlashAttentionBackend(DecodeOnlyAttentionBackend):

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "flash-attn"

    @staticmethod
    def get_impl_cls() -> Type["DecodeOnlyFlashAttentionImpl"]:
        return DecodeOnlyFlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["DecodeOnlyFlashAttentionMetadata"]:
        return DecodeOnlyFlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["DecodeOnlyFlashAttentionMetadataBuilder"]:
        return DecodeOnlyFlashAttentionMetadataBuilder


@dataclass
class DecodeOnlyFlashAttentionMetadata(DecodeOnlyAttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    _cached_prefill_metadata: Optional[
        "DecodeOnlyFlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional[
        "DecodeOnlyFlashAttentionMetadata"] = None

    @property
    def prefill_metadata(self) -> Optional["DecodeOnlyFlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        self._cached_prefill_metadata = DecodeOnlyFlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables)
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["DecodeOnlyFlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.seq_lens_tensor is not None

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        self._cached_decode_metadata = DecodeOnlyFlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens=None,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(self.query_start_loc[self.num_prefills:] -
                             self.query_start_loc[self.num_prefills])
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            context_lens_tensor=None,
            block_tables=block_tables)
        return self._cached_decode_metadata

    def to(self, device, non_blocking=True):
        for k in self.__dict__:
            if not hasattr(self.__dict__[k], "to"):
                continue
            self.__dict__[k] = self.__dict__[k].to(device=device,
                                                   non_blocking=non_blocking)

        return self


class DecodeOnlyFlashAttentionMetadataBuilder(
        DecodeOnlyAttentionMetadataBuilder[DecodeOnlyFlashAttentionMetadata]):

    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, scheduled_requests: List[DecodingSchedulableRequest]):
        seq_lens = [request.c_seq_len for request in scheduled_requests]

        block_tables = make_tensor_with_pad(
            [request.c_physical_block_ids for request in scheduled_requests],
            pad=0,
            dtype=torch.int,
            device="cpu",
            pin_memory=pin_memory)

        context_lens_tensor = torch.tensor(
            [request.c_context_len for request in scheduled_requests],
            dtype=torch.int,
            device="cpu",
            pin_memory=pin_memory)

        seq_lens_tensor = torch.tensor(seq_lens,
                                       dtype=torch.int,
                                       device="cpu",
                                       pin_memory=pin_memory)

        query_lens_tensor = torch.tensor(
            [request.c_query_len for request in scheduled_requests],
            dtype=torch.long,
            device="cpu",
            pin_memory=pin_memory)

        query_start_loc_tensor = torch.zeros(query_lens_tensor.shape[0] + 1,
                                             dtype=torch.int32,
                                             device="cpu",
                                             pin_memory=pin_memory)
        seq_start_loc_tensor = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                           dtype=torch.int32,
                                           device="cpu",
                                           pin_memory=pin_memory)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc_tensor.dtype,
                     out=seq_start_loc_tensor[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc_tensor.dtype,
                     out=query_start_loc_tensor[1:])

        slot_mapping = []
        for request in scheduled_requests:
            # Compute slot mapping.
            is_profile_run = block_tables.nelement() == 0
            start_idx = 0

            compute_slot_mapping(is_profile_run, slot_mapping,
                                 request.c_seq_len, request.c_context_len,
                                 start_idx, self.block_size,
                                 request.c_physical_block_ids)

        slot_mapping_tensor = torch.tensor(slot_mapping,
                                           dtype=torch.long,
                                           device="cpu",
                                           pin_memory=pin_memory)

        max_query_len = max(
            [request.c_query_len for request in scheduled_requests])

        max_prefill_seq_len = max([
            request.c_seq_len
            for request in scheduled_requests if request.c_is_prefill
        ],
                                  default=0)

        num_prefills = sum(1 for request in scheduled_requests
                           if request.c_is_prefill)
        num_prefill_tokens = sum([
            request.token_chunk_size for request in scheduled_requests
            if request.c_is_prefill
        ])

        max_decode_seq_len = max([
            request.c_seq_len
            for request in scheduled_requests if not request.c_is_prefill
        ],
                                 default=0)

        max_decode_query_len = max([
            request.c_query_len
            for request in scheduled_requests if not request.c_is_prefill
        ],
                                   default=0)

        num_decode_tokens = sum(1 for request in scheduled_requests
                                if not request.c_is_prefill)

        return DecodeOnlyFlashAttentionMetadata(
            num_prefills=num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
        )


class DecodeOnlyFlashAttentionImpl(DecodeOnlyAttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window - 1,
                                0) if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None)

        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = DecodeOnlyFlashAttentionBackend.get_supported_head_sizes(
        )
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")

        self._q_scale = torch.tensor(1.0, dtype=torch.float32)
        self._k_scale = torch.tensor(1.0, dtype=torch.float32)
        self._v_scale = torch.tensor(1.0, dtype=torch.float32)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: DecodeOnlyFlashAttentionMetadata,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be None for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """

        output_shape = query.shape

        output = torch.empty(output_shape,
                             dtype=query.dtype,
                             device=query.device)

        hidden_size = output_shape[-1]

        if kv_cache is None:
            return output

        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size)

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap

        if kv_cache is not None:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]

            updated_slot_mapping = attn_metadata.slot_mapping

            # Reshape the input keys and values and store them in the cache.
            # If kv_cache is not provided, the new key and value tensors are
            # not cached. This happens during the initial memory
            # profiling run.
            torch.ops._C_cache_ops.reshape_and_cache_flash(
                key,
                value,
                kv_cache[0],
                kv_cache[1],
                updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                kv_cache_dtype,
                self._k_scale,
                self._v_scale,
            )

        num_prefill_query_tokens = attn_metadata.num_prefill_tokens
        num_prefill_kv_tokens = attn_metadata.num_prefill_tokens
        num_decode_query_tokens = attn_metadata.num_decode_tokens

        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache is None or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.

                q_seq_start_loc = attn_metadata.seq_start_loc
                q_seq_len = attn_metadata.max_prefill_seq_len
                k_seq_start_loc = attn_metadata.seq_start_loc
                k_seq_len = attn_metadata.max_prefill_seq_len

                key = key[:num_prefill_kv_tokens]
                value = value[:num_prefill_kv_tokens]

                descale_shape = (q_seq_start_loc.shape[0] - 1, key.shape[1])
                flash_attn_varlen_func(
                    q=query,
                    k=key,
                    v=value,
                    cu_seqlens_q=q_seq_start_loc,
                    cu_seqlens_k=k_seq_start_loc,
                    max_seqlen_q=q_seq_len,
                    max_seqlen_k=k_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=self._q_scale.expand(descale_shape),
                    k_descale=self._k_scale.expand(descale_shape),
                    v_descale=self._v_scale.expand(descale_shape),
                )
            else:
                # prefix-enabled attention
                assert prefill_meta.seq_lens is not None
                assert prefill_meta.query_start_loc is not None
                max_seq_len = max(prefill_meta.seq_lens)
                descale_shape = (prefill_meta.query_start_loc.shape[0] - 1,
                                 key.shape[1])
                flash_attn_varlen_func(  # noqa
                    q=query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=prefill_meta.query_start_loc,
                    max_seqlen_q=prefill_meta.max_query_len,
                    seqused_k=prefill_meta.seq_lens_tensor,
                    max_seqlen_k=max_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    block_table=prefill_meta.block_tables,
                    softcap=logits_soft_cap,
                    out=prefill_output,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=self._q_scale.expand(descale_shape),
                    k_descale=self._k_scale.expand(descale_shape),
                    v_descale=self._v_scale.expand(descale_shape),
                )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.

            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert decode_meta.query_start_loc is not None
                descale_shape = (decode_meta.query_start_loc.shape[0] - 1,
                                 key.shape[1])
                flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    seqused_k=decode_meta.seq_lens_tensor,
                    max_seqlen_k=decode_meta.max_decode_seq_len,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    block_table=decode_meta.block_tables,
                    out=decode_output,
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=self._q_scale.expand(descale_shape),
                    k_descale=self._k_scale.expand(descale_shape),
                    v_descale=self._v_scale.expand(descale_shape),
                )
            else:
                # Use flash_attn_with_kvcache for normal decoding.
                seq_lens = decode_meta.seq_lens_tensor
                block_tables = decode_meta.block_tables
                descale_shape = (block_tables.shape[0], key_cache.shape[-2])

                flash_attn_with_kvcache(
                    q=decode_query.unsqueeze(1),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    block_table=block_tables,
                    cache_seqlens=seq_lens,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=window_size,
                    alibi_slopes=alibi_slopes,
                    softcap=logits_soft_cap,
                    out=decode_output.unsqueeze(1),
                    fa_version=self.vllm_flash_attn_version,
                    q_descale=self._q_scale.expand(descale_shape),
                    k_descale=self._k_scale.expand(descale_shape),
                    v_descale=self._v_scale.expand(descale_shape),
                )
        return output.view(-1, hidden_size)
