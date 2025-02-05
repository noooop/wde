import inspect

from wde.logger import init_logger
from wde.microservices.framework.nameserver.client import ZeroClient
from wde.workflows.decoding.kv_cache.remote.schema import (
    ContainsRequest, ContainsResponse, GetRequest, GetResponse,
    GetResponseStream, InfoResponse, SetRequest, SetResponse)

logger = init_logger(__name__)

CLIENT_VALIDATION = True


class ZeroRemoteKVCacheClient(ZeroClient):
    protocol = "remote_kv_cache"

    def get(self, name, model, block_hashs, stream=False):
        method = "get"

        data = {"model": model, "block_hashs": block_hashs, "stream": stream}

        if CLIENT_VALIDATION:
            data = GetRequest(**data).dict()

        response = self.query(name, method, data)

        if not inspect.isgenerator(response):
            return self.check_response(name, response, GetResponse)
        else:

            def generator():
                for rep in response:
                    yield self.check_response(name, rep, GetResponseStream)

            return generator()

    def set(self,
            name,
            model,
            block_hashs,
            blocks,
            force=False,
            deferred=True):
        method = "set"

        data = {
            "model": model,
            "block_hashs": block_hashs,
            "blocks": blocks,
            "force": force,
            "deferred": deferred
        }

        if CLIENT_VALIDATION:
            data = SetRequest(**data).dict()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, SetResponse)

    def contains(self, name, model, block_hashs, refresh=False):
        method = "contains"

        data = {"model": model, "block_hashs": block_hashs, "refresh": refresh}

        if CLIENT_VALIDATION:
            data = ContainsRequest(**data).dict()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, ContainsResponse)

    def info(self, name):
        method = "info"

        rep = self.query(name, method)
        return self.check_response(name, rep, InfoResponse)
