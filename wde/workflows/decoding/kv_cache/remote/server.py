import numpy as np
from gevent.threadpool import ThreadPoolExecutor

from wde.logger import init_logger
from wde.microservices.framework.zero.schema import ZeroServerResponseOk
from wde.microservices.framework.zero.server import Z_MethodZeroServer
from wde.workflows.decoding.kv_cache.remote.memory import RemoteMemoryKVCache
from wde.workflows.decoding.kv_cache.remote.schema import (
    ContainsRequest, ContainsResponse, GetRequest, GetResponse, SetRequest,
    SetResponse)

logger = init_logger(__name__)


class ZeroRemoteKVCacheServer(Z_MethodZeroServer):
    protocol = "remote_kv_cache"

    def __init__(self, name, model, engine_args, **kwargs):
        super().__init__(name=name, port=None, do_register=True, **kwargs)
        self._cache = None
        self.model = model
        self.engine_args = engine_args
        self.executor = ThreadPoolExecutor(1)

    def init(self):
        self._cache = RemoteMemoryKVCache(model=self.model, **self.engine_args)
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

        total = len(request.block_hashs)

        block_hashs = []
        blocks = []

        for i in range(total):
            block_hash = request.block_hashs[i]

            block = self._cache.get(block_hash)

            if block is None:
                continue

            block_hashs.append(block_hash)
            blocks.append(block)

        block_hashs = np.array(block_hashs, dtype=request.block_hashs.dtype)
        rep = ZeroServerResponseOk(
            msg=GetResponse(block_hashs=block_hashs, blocks=blocks).dict())
        self.zero_send(req, rep)

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

        force = request.force

        total = len(request.block_hashs)
        exist = 0
        for i in range(total):
            block_hash = request.block_hashs[i]
            data = request.blocks[i]

            if not force and block_hash in self._cache:
                exist += 1
                continue

            self._cache.set(block_hash, data)

        rep = ZeroServerResponseOk(
            msg=SetResponse(total=total, exist=exist).dict())
        self.zero_send(req, rep)

    def z_contains(self, req):
        request = ContainsRequest(**req.data)

        if request.model != self.model:
            self.handle_error(
                req=req, err_msg=f"model [{request.model}] not supported!")

        total = len(request.block_hashs)
        block_hashs = []

        for i in range(total):
            block_hash = request.block_hashs[i]
            if block_hash in self._cache:
                block_hashs.append(block_hash)

        block_hashs = np.array(block_hashs, dtype=request.block_hashs.dtype)
        rep = ZeroServerResponseOk(msg=ContainsResponse(
            block_hashs=block_hashs).dict())
        self.zero_send(req, rep)
