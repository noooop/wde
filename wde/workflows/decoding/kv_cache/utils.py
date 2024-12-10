import numpy as np


class RefCounter:

    def __init__(self, num_total_blocks):
        self.num_total_blocks = num_total_blocks
        self._counter = np.zeros(self.num_total_blocks, dtype=np.int64)

    def incr(self, block_id):
        assert block_id < self.num_total_blocks
        self._counter[block_id] += 1

    def decr(self, block_id):
        self._counter[block_id] -= 1

        return self._counter[block_id]
