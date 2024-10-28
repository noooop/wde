from wde.microservices.framework.nameserver.async_client import AsyncZeroClient
from wde.microservices.framework.nameserver.client import ZeroClient
from wde.tasks.reranker.engine.schema import (PROTOCOL, RerankerRequest,
                                              RerankerResponse)

CLIENT_VALIDATION = True


class RerankerClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def compute_score(self, name, pairs, options=None):
        method = "compute_score"
        data = {"model": name, "pairs": pairs, "options": options or dict()}

        if CLIENT_VALIDATION:
            data = RerankerRequest(**data).model_dump()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, RerankerResponse)


class AsyncRerankerClient(AsyncZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        AsyncZeroClient.__init__(self, self.protocol, nameserver_port)

    async def compute_score(self, name, pairs, options=None):
        method = "compute_score"
        data = {"model": name, "pairs": pairs, "options": options or dict()}

        if CLIENT_VALIDATION:
            data = RerankerRequest(**data).model_dump()

        rep = await self.query(name, method, data)
        return self.check_response(name, rep, RerankerResponse)
