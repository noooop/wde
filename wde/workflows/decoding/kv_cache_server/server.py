import numpy as np
from gevent.threadpool import ThreadPoolExecutor

from wde.logger import init_logger
from wde.microservices.framework.zero.schema import (ZeroServerRequest,
                                                     ZeroServerResponseOk,
                                                     ZeroServerStreamResponseOk
                                                     )
from wde.microservices.framework.zero.server import Z_MethodZeroServer
from wde.workflows.decoding.kv_cache_server.memory import RemoteMemoryKVCache
from wde.workflows.decoding.kv_cache_server.schema import (
    ContainsRequest, ContainsResponse, GetRequest, GetResponse,
    GetResponseStream, InfoResponse, SetRequest, SetResponse)

logger = init_logger(__name__)


class ZeroRemoteKVCacheServer(Z_MethodZeroServer):
    protocol = "remote_kv_cache"

    def __init__(self,
                 model,
                 block_size,
                 memory_space,
                 cache_dtype="auto",
                 name=None,
                 max_workers=4,
                 **kwargs):

        if name is None:
            name = f"kv_cache:{model}:{block_size}"

        super().__init__(name=name, port=None, do_register=True, **kwargs)

        self._cache = None
        self.model = model
        self.engine_args = {
            "model": model,
            "block_size": block_size,
            "memory_space": memory_space,
            "cache_dtype": cache_dtype
        }
        self.threads = ThreadPoolExecutor(max_workers)

    def init(self):
        self._cache = RemoteMemoryKVCache(**self.engine_args)
        logger.info("%s for %s running! port: %d", self.__class__.__name__,
                    self.name, self.port)

    def clean_up(self):
        self._cache = None
        super().clean_up()

    def z_get(self, req):
        request = GetRequest(**req.data)

        if request.model != self.model:
            self.handle_error(
                req=req, err_msg=f"model [{request.model}] not supported!")

        stream = request.stream
        total = len(request.block_hashs)

        hit = 0
        miss = 0

        blocks = []
        block_hashs = []
        block_hashs_set = set()
        for i in range(total):
            block_hash = request.block_hashs[i]

            if block_hash in block_hashs_set:
                continue

            block = self._cache.get(block_hash)

            if block is None:
                miss += 1
                continue

            self._cache.block_allocator.hold(block)

            hit += 1

            block_hashs.append(block_hash)
            block_hashs_set.add(block_hash)
            blocks.append(block)

        def send():
            block_hashs_np = np.array(block_hashs,
                                      dtype=request.block_hashs.dtype)

            if stream:
                rep = ZeroServerStreamResponseOk(rep_id=0,
                                                 snd_more=hit > 0,
                                                 msg=GetResponse(
                                                     total=total,
                                                     hit=hit,
                                                     miss=miss).dict())

                self.zero_send(req, rep)

                for i in range(hit):
                    block_hash = block_hashs_np[i:i + 1]
                    data = self._cache.kv_cache[blocks[i].physical_block_id]

                    rep = ZeroServerStreamResponseOk(rep_id=i + 1,
                                                     snd_more=not i + 1 == hit,
                                                     msg=GetResponseStream(
                                                         block_hash=block_hash,
                                                         block=data).dict())
                    self.zero_send(req, rep)

            else:
                data = [
                    self._cache.kv_cache[block.physical_block_id]
                    for block in blocks
                ]

                rep = ZeroServerResponseOk(
                    msg=GetResponse(total=total,
                                    hit=hit,
                                    miss=miss,
                                    block_hashs=block_hashs_np,
                                    blocks=data).dict())
                self.zero_send(req, rep)

        send()

        for block in blocks:
            self._cache.block_allocator.free(block)

    def z_set(self, req):
        request = SetRequest(**req.data)

        if request.model != self.model:
            self.handle_error(
                req=req, err_msg=f"model [{request.model}] not supported!")

        if len(request.block_hashs) != len(request.blocks):
            self.handle_error(req=req,
                              err_msg=f"len(request.block_hashs) "
                              f"[{len(request.block_hashs)}] != "
                              f"len(request.blocks)! [{len(request.blocks)}]")

        block_shape = self._cache.block_shape

        force = request.force
        deferred = request.deferred
        total = len(request.block_hashs)
        exist = 0

        blocks = []
        for i in range(total):
            data = request.blocks[i]

            assert data.shape == block_shape

            block_hash = request.block_hashs[i]

            block = self._cache.get_or_create(block_hash)

            if block is None:
                # NoFreeBlocksError
                continue

            if block.lock:
                # doing write
                continue

            if block.lock is not None and not force:
                # has been written
                exist += 1
                continue

            block.acquire()

            self._cache.block_allocator.hold(block)
            blocks.append((block, data))

        def memcpy():
            for block, data in blocks:
                self._cache.kv_cache[block.physical_block_id] = data

        if deferred:
            rep = ZeroServerResponseOk(
                msg=SetResponse(total=total, exist=exist).dict())
            self.zero_send(req, rep)

            f = self.threads.submit(memcpy)
            f.result()

            for block, data in blocks:
                block.release()
                self._cache.block_allocator.free(block)
        else:
            f = self.threads.submit(memcpy)
            f.result()

            for block, data in blocks:
                block.release()
                self._cache.block_allocator.free(block)

            rep = ZeroServerResponseOk(
                msg=SetResponse(total=total, exist=exist).dict())
            self.zero_send(req, rep)

    def z_contains(self, req):
        request = ContainsRequest(**req.data)

        if request.model != self.model:
            self.handle_error(
                req=req, err_msg=f"model [{request.model}] not supported!")

        refresh = request.refresh
        total = len(request.block_hashs)
        dtype = request.block_hashs.dtype

        hit = []
        miss = []

        for i in range(total):
            block_hash = request.block_hashs[i]

            if self._cache.contains(block_hash, refresh):
                hit.append(block_hash)
            else:
                miss.append(block_hash)

        hit = np.array(hit, dtype=dtype)
        miss = np.array(miss, dtype=dtype)

        rep = ZeroServerResponseOk(
            msg=ContainsResponse(hit=hit, miss=miss).dict())
        self.zero_send(req, rep)

    def z_info(self, req: ZeroServerRequest):
        rep = ZeroServerResponseOk(msg=InfoResponse(**self._cache.info))
        self.zero_send(req, rep)
