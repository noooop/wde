from typing import List

PAD_SLOT_ID = -1


def compute_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
                         seq_len: int, context_len: int, start_idx: int,
                         block_size: int, block_table: List[int]):
    if is_profile_run:
        slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        return

    slot_mapping.extend([PAD_SLOT_ID] * max(0, start_idx - context_len))
    for i in range(max(start_idx, context_len), seq_len):
        block_number = block_table[i // block_size]
        block_offset = i % block_size
        slot = block_number * block_size + block_offset
        slot_mapping.append(slot)
