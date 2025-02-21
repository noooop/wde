from hashlib import md5
from typing import List, Optional, Union

import numpy as np

PrefixHash = Optional[bytes]
TokenIDs = Union[List[int], tuple[int, ...]]
PrefixHashDtype = "S1"
PrefixHashLen = 32


def get_prefix_hash(b: bytes):
    return md5(b).hexdigest().encode("utf-8")


def get_block_hash(prefix_hash: PrefixHash, delta_token_ids: TokenIDs):
    return get_prefix_hash(prefix_hash +
                           np.array(delta_token_ids, dtype=np.int64).tobytes())


def block_hashs_to_numpy_array(block_hashs):
    n = len(block_hashs)

    if n == 0:
        return np.empty((0, PrefixHashLen), dtype=PrefixHashDtype)

    assert PrefixHashLen == len(block_hashs[0])

    array = np.empty((n, PrefixHashLen), dtype=PrefixHashDtype)

    for i in range(n):
        array[i] = np.frombuffer(block_hashs[i], dtype=PrefixHashDtype)

    return array
