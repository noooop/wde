import gc

import torch

from wde.engine.gevent_engine import GeventLLMEngine
from wde.logger import init_logger
from wde.microservices.framework.zero.schema import ZeroServerResponseOk
from wde.microservices.framework.zero.server import Z_MethodZeroServer
from wde.tasks.retriever.engine.schema import (RetrieverRequest,
                                               RetrieverResponse)

logger = init_logger(__name__)


class ZeroEngine(Z_MethodZeroServer):

    def __init__(self, name, engine_args, **kwargs):
        self.engine_args = engine_args
        self.engine = GeventLLMEngine(**self.engine_args)

        Z_MethodZeroServer.__init__(
            self,
            name=name,
            protocol=self.engine.engine.workflow.protocol,
            port=None,
            do_register=True,
            **kwargs)

    def init(self):
        logger.info("%s %s is running! port: %d", self.__class__.__name__,
                    self.name, self.port)

    def z_info(self, req):
        rep = ZeroServerResponseOk(msg=self.engine.info)
        self.zero_send(req, rep)

    def __del__(self):
        try:
            self.engine.terminate()
            self.engine = None

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def z_encode(self, req):
        request = RetrieverRequest(**req.data)
        outputs = self.engine.encode(inputs=request.inputs,
                                     request_id=str(req.req_id))

        output = list(outputs)[0]

        response = RetrieverResponse(model=request.model,
                                     embedding=output.outputs.numpy())

        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


def start_zero_engine(engine_args):
    assert "model" in engine_args

    from wde.microservices.framework.zero.server import ZeroServerProcess
    from wde.microservices.framework.zero_manager.client import \
        ZeroManagerClient

    MANAGER_NAME = "RootZeroManager"
    server_class = "wde.engine.zero_engine:ZeroEngine"

    nameserver = ZeroServerProcess(
        "wde.microservices.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess(
        "wde.microservices.framework.zero_manager.server:ZeroManager",
        server_kwargs={
            "name":
            MANAGER_NAME,
            "server_class":
            "zerollama.core.framework.zero_manager.server:ZeroManager"
        })

    nameserver.start()
    manager.start()

    manager_client = ZeroManagerClient(MANAGER_NAME)
    manager_client.wait_service_available(MANAGER_NAME)

    model_name = engine_args["model"]

    manager_client.start(name=model_name,
                         engine_kwargs={
                             "server_class": server_class,
                             "engine_args": engine_args
                         })

    return manager, nameserver
