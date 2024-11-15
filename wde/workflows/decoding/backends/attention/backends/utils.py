"""Attention backend utils"""
from typing import Dict, List, Union

STR_NOT_IMPL_ENC_DEC_ROCM_HIP = ("ROCm/HIP is not currently supported "
                                 "with encoder/decoder models.")

PAD_SLOT_ID = -1


def is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    if isinstance(block_tables, dict) and all(
            value is None for value in block_tables.values()):
        return True
    return False


def compute_slot_mapping_start_idx(is_prompt: bool, query_len: int,
                                   context_len: int, sliding_window: int):
    """
    Compute the start index of slot mapping.
    """
    start_idx = 0
    if is_prompt and sliding_window is not None:
        start_idx = max(0, query_len - sliding_window)
    return start_idx


def compute_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
                         seq_id: int, seq_len: int, context_len: int,
                         start_idx: int, block_size: int,
                         block_tables: Dict[int, List[int]]):
    """
    Compute slot mapping.
    """
    if is_profile_run:
        # During memory profiling, the block tables are not
        # initialized yet. In this case, we just use a dummy
        # slot mapping.
        # In embeddings, the block tables are {seq_id: None}.
        slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        return

    # Mask the [0, start_idx) tokens of the prompt with
    # PAD_SLOT_ID, where start_idx is max(0, seq_len -
    # sliding_window). For example, if the prompt len is 10,
    # sliding window is 8, and block size is 4, the first two
    # tokens are masked and the slot mapping will be
    # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    block_table = block_tables[seq_id]
    slot_mapping.extend([PAD_SLOT_ID] * max(0, start_idx - context_len))
    for i in range(max(start_idx, context_len), seq_len):
        block_number = block_table[i // block_size]
        block_offset = i % block_size
        slot = block_number * block_size + block_offset
        slot_mapping.append(slot)