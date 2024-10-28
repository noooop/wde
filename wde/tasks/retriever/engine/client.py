from wde.microservices.framework.nameserver.client import ZeroClient
from wde.tasks.retriever.engine.schema import (PROTOCOL, RetrieverRequest,
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
