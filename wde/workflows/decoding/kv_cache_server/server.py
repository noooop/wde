import numpy as np
from gevent.threadpool import ThreadPoolExecutor

from wde.logger import init_logger
from wde.microservices.framework.zero.schema import (ZeroServerRequest,
                                                     ZeroServerResponseOk,
                                                     ZeroServerStreamResponseOk
                                                     )
from wde.microservices.framework.zero.server import Z_MethodZeroServer
from wde.utils import lazy_import
from wde.workflows.decoding.kv_cache_server.Interface import \
    RemoteKVCacheInterface
from wde.workflows.decoding.kv_cache_server.schema import (
    ContainsRequest, ContainsResponse, GetRequest, GetResponse,
    GetResponseStream, InfoResponse, SetRequest, SetResponse)

logger = init_logger(__name__)


class ZeroRemoteKVCacheServer(Z_MethodZeroServer):
    protocol = "remote_kv_cache"

    RemoteMemoryKVCacheClass = "wde.workflows.decoding.kv_cache_server.memory:RemoteMemoryKVCache"
    RemoteFilesystemKVCache = "wde.workflows.decoding.kv_cache_server.filesystem:RemoteFilesystemKVCache"
    RemoteHybridKVCacheCache = "wde.workflows.decoding.kv_cache_server.hybrid:RemoteHybridKVCache"

    def __init__(self, engine_args, **kwargs):
        model = engine_args["model"]
        block_size = engine_args["block_size"]
        name = kwargs.pop("name", None) or f"kv_cache:{model}:{block_size}"
        max_workers = engine_args.pop("max_workers", 4)

        super().__init__(name=name, do_register=True, **kwargs)

        self.model = model
        self.memory_space = engine_args.pop("memory_space", None)
        self.file_space = engine_args.pop("file_space", None)
        self.kv_cache_folder = engine_args.pop("kv_cache_folder", None)
        self.block_size = block_size
        self.cache_dtype = engine_args.pop("cache_dtype", "auto")
        self._cache: RemoteKVCacheInterface
        self.threads = ThreadPoolExecutor(max_workers)

    def init(self):

        def validate(n):
            return isinstance(n, (float, int)) and n > 0.001

        if validate(self.memory_space) and not validate(self.file_space):
            remote_kv_cache_class = self.RemoteMemoryKVCacheClass
            logger.info("remote_kv_cache use RemoteMemoryKVCache")
        elif validate(self.file_space) and isinstance(
                self.kv_cache_folder, str) and not validate(self.memory_space):
            remote_kv_cache_class = self.RemoteFilesystemKVCache
            logger.info("remote_kv_cache use RemoteFilesystemKVCache")
        elif validate(self.file_space) and isinstance(
                self.kv_cache_folder, str) and validate(self.memory_space):
            remote_kv_cache_class = self.RemoteHybridKVCacheCache
            logger.info("remote_kv_cache use RemoteHybridKVCacheCache")
        else:
            raise ValueError("remote kv cache config not supported")

        self._cache = lazy_import(remote_kv_cache_class)(
            model=self.model,
            memory_space=self.memory_space,
            file_space=self.file_space,
            kv_cache_folder=self.kv_cache_folder,
            block_size=self.block_size,
            cache_dtype=self.cache_dtype)

        self._cache.init()

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

        info, callback, release = self._cache.get(request.block_hashs)

        info = GetResponse(**info)

        if stream:
            rep = ZeroServerStreamResponseOk(rep_id=0,
                                             snd_more=info.hit > 0,
                                             msg=info.dict())
            self.zero_send(req, rep)

            for i, (block_hash, data) in enumerate(callback()):
                rep = ZeroServerStreamResponseOk(
                    rep_id=i + 1,
                    snd_more=not i + 1 == info.hit,
                    msg=GetResponseStream(block_hash=block_hash,
                                          block=data).dict())
                self.zero_send(req, rep)
        else:
            block_data = []
            block_hashs = []
            for block_hash, data in callback():
                block_hashs.append(block_hash)
                block_data.append(data)

            block_hashs_np = np.array(block_hashs,
                                      dtype=request.block_hashs.dtype)

            info.block_hashs = block_hashs_np
            info.blocks = block_data

            rep = ZeroServerResponseOk(msg=info.dict())
            self.zero_send(req, rep)

        release()

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

        info, generator, release = self._cache.set(
            block_hashs=request.block_hashs,
            block_data=request.blocks,
            force=request.force)

        if request.deferred:
            rep = ZeroServerResponseOk(msg=SetResponse(**info).dict())
            self.zero_send(req, rep)

            f = self.threads.submit(generator)
            f.result()

            release()
        else:
            f = self.threads.submit(generator)
            f.result()

            release()

            rep = ZeroServerResponseOk(msg=SetResponse(**info).dict())
            self.zero_send(req, rep)

    def z_contains(self, req):
        request = ContainsRequest(**req.data)

        if request.model != self.model:
            self.handle_error(
                req=req, err_msg=f"model [{request.model}] not supported!")

        hit, miss = self._cache.contains(request.block_hashs, request.refresh)

        rep = ZeroServerResponseOk(
            msg=ContainsResponse(hit=hit, miss=miss).dict())
        self.zero_send(req, rep)

    def z_info(self, req: ZeroServerRequest):
        rep = ZeroServerResponseOk(msg=InfoResponse(**self._cache.info))
        self.zero_send(req, rep)
