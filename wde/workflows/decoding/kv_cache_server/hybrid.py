import numpy as np

from wde.workflows.decoding.kv_cache_server.filesystem import \
    RemoteFilesystemKVCache
from wde.workflows.decoding.kv_cache_server.Interface import \
    RemoteKVCacheInterface
from wde.workflows.decoding.kv_cache_server.memory import RemoteMemoryKVCache


class RemoteHybridKVCache(RemoteKVCacheInterface):

    def __init__(self, *args, **kwargs):
        self._memory_cache = RemoteMemoryKVCache(*args, **kwargs)
        self._persistence_cache = RemoteFilesystemKVCache(*args, **kwargs)

    def init(self):
        self._memory_cache.init()
        self._persistence_cache.init()

    def set(self, block_hashs, block_data, force):
        m_info, m_generator, m_release = self._memory_cache.set(
            block_hashs, block_data, force)
        p_info, p_generator, p_release = self._persistence_cache.set(
            block_hashs, block_data, force)

        def generator():
            m_generator()
            p_generator()

        def release():
            m_release()
            p_release()

        return p_info, generator, release

    def contains(self, block_hashs, refresh):
        m_hit, m_miss = self._memory_cache.contains(block_hashs, refresh)
        p_hit, p_miss = self._persistence_cache.contains(m_miss, refresh)

        hit = np.hstack([m_hit, p_hit])
        return hit, p_miss

    def get(self, block_hashs):
        m_block_allocator = self._memory_cache.block_allocator
        p_block_allocator = self._persistence_cache.block_allocator
        get_s_block_hash = self._persistence_cache.get_s_block_hash

        block_shape = self._persistence_cache.block_shape
        dtype = self._persistence_cache.dtype

        total = len(block_hashs)
        hit = 0
        miss = 0
        duplicate = 0

        blocks = {}
        for i in range(total):
            block_hash = block_hashs[i]

            if block_hash in blocks:
                duplicate += 1
                continue

            m_block = m_block_allocator.get(block_hash)

            if m_block is not None:
                # hit in memory
                hit += 1

                m_block_allocator.hold(m_block)
                blocks[block_hash] = (m_block, i)
                continue

            s_block_hash = get_s_block_hash(block_hashs, i)
            p_block = p_block_allocator.get(s_block_hash)

            if p_block is not None:
                # hit in persistence_cache
                hit += 1
                p_block_allocator.hold(p_block)

                # write data to memory
                m_block = m_block_allocator.get_or_create(block_hash)

                if m_block.lock:
                    m_block = None

                if m_block is not None:
                    m_block.acquire()
                    m_block_allocator.hold(m_block)

                blocks[block_hash] = (p_block, i, s_block_hash, m_block)

                continue

            miss += 1

        def generator():
            for items in blocks.values():
                if len(items) == 2:
                    # hit in memory
                    m_block, index = items

                    block_hash = block_hashs[index:index + 1]
                    data = self._memory_cache.kv_cache[
                        m_block.physical_block_id]
                    yield block_hash, data
                else:
                    # hit in persistence_cache
                    p_block, index, s_block_hash, m_block = items
                    block_hash = block_hashs[index:index + 1]

                    directory = (self._persistence_cache.kv_cache_folder /
                                 s_block_hash[:2] / s_block_hash[2:4])

                    data = np.load(directory / (s_block_hash + ".npy"))

                    assert data.shape == block_shape
                    assert data.dtype == dtype

                    self._memory_cache.kv_cache[
                        m_block.physical_block_id] = data

                    yield block_hash, data

        def release():
            for items in blocks.values():
                if len(items) == 2:
                    m_block, index = items
                    m_block_allocator.free(m_block)
                else:
                    p_block, index, s_block_hash, m_block = items
                    p_block_allocator.free(p_block)

                    m_block.release()
                    m_block_allocator.free(m_block)

        info = {
            "total": total,
            "hit": hit,
            "miss": miss,
            "duplicate": duplicate,
        }

        return info, generator, release

    def __contains__(self, block_hash):
        return block_hash in self._memory_cache or block_hash in self._persistence_cache

    def __len__(self):
        return len(self._persistence_cache)

    @property
    def info(self):
        return self._persistence_cache.info
