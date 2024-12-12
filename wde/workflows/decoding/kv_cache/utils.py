def chunk_list(lst, chunk_size: int):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def get_num_required_blocks(num_token_ids: int,
                            block_size: int,
                            num_lookahead_slots: int = 0) -> int:

    def cdiv(a: int, b: int) -> int:
        return -(a // -b)

    return cdiv(num_token_ids + num_lookahead_slots, block_size)
