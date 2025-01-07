from queue import Queue

import torch


class StreamPool:

    def __init__(self):
        self.pool = Queue()

    def get(self):
        if self.pool.empty():
            stream = torch.cuda.Stream()
            return stream
        else:
            return self.pool.get()

    def put(self, stream):
        return self.pool.put(stream)
