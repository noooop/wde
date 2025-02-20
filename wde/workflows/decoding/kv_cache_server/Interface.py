class RemoteKVCacheInterface:

    def init(self):
        raise NotImplementedError

    def set(self, block_hashs, block_data, force):
        raise NotImplementedError

    def contains(self, block_hashs, refresh):
        raise NotImplementedError

    def get(self, block_hashs):
        raise NotImplementedError

    def __contains__(self, block_hash):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    @property
    def info(self):
        raise NotImplementedError
