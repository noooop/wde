from wde.microservices.framework.nameserver.async_client import AsyncZeroClient
from wde.microservices.framework.nameserver.client import ZeroClient
from wde.tasks.retriever.schema.api import (PROTOCOL, RetrieverRequest,
                                            RetrieverResponse)

CLIENT_VALIDATION = True


class RetrieverClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def encode(self, name, inputs, options=None):
        method = "encode"
        data = {"model": name, "inputs": inputs, "options": options or dict()}

        if CLIENT_VALIDATION:
            data = RetrieverRequest(**data).model_dump()

        rep = self.query(name, method, data)
        return self.check_response(name, rep, RetrieverResponse)


class AsyncRetrieverClient(AsyncZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        AsyncZeroClient.__init__(self, self.protocol, nameserver_port)

    async def encode(self, name, inputs, options=None):
        method = "encode"
        data = {"model": name, "inputs": inputs, "options": options or dict()}

        if CLIENT_VALIDATION:
            data = RetrieverRequest(**data).model_dump()

        rep = await self.query(name, method, data)
        return self.check_response(name, rep, RetrieverResponse)
