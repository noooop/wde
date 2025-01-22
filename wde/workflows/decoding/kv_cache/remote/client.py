from wde.logger import init_logger
from wde.microservices.framework.nameserver.client import ZeroClient
from wde.workflows.decoding.kv_cache.remote.schema import (
    ContainsRequest, ContainsResponse, GetRequest, GetResponse, SetRequest,
    SetResponse)

logger = init_logger(__name__)

CLIENT_VALIDATION = True


class ZeroRemoteKVCacheClient(ZeroClient):
    protocol = "remote_kv_cache"

    def get(self, name, model, block_hashs):
        method = "get"

        data = {
            "model": model,
            "block_hashs": block_hashs,
        }

        if CLIENT_VALIDATION:
            data = GetRequest(**data).dict()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, GetResponse)

    def set(self, name, model, block_hashs, blocks, force=False):
        method = "set"

        data = {
            "model": model,
            "block_hashs": block_hashs,
            "blocks": blocks,
            "force": force
        }

        if CLIENT_VALIDATION:
            data = SetRequest(**data).dict()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, SetResponse)

    def contains(self, name, model, block_hashs):
        method = "contains"

        data = {
            "model": model,
            "block_hashs": block_hashs,
        }

        if CLIENT_VALIDATION:
            data = ContainsRequest(**data).dict()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, ContainsResponse)
