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
        self.return_metrics = self.engine_args.pop("return_metrics", None)

        Z_MethodZeroServer.__init__(
            self,
            name=name,
            protocol=self.engine.engine.workflow.protocol,
            port=None,
            do_register=True,
            pool_size=self.engine.engine.engine_config.scheduler_config.max_num_requests * 4,
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

        if self.return_metrics:
            m = output.metrics

            metrics = {
                "waiting_time": m.waiting_time,
                "scheduler_time": m.scheduler_time,
                "n_request_in_batch": m.n_request_in_batch,
                "waiting4execution": m.waiting4execution,
                "execute_time": m.execute_time,
                "delay": m.delay
            }
        else:
            metrics = None

        response = RetrieverResponse(model=request.model,
                                     embedding=output.outputs.numpy(),
                                     metrics=metrics)

        rep = ZeroServerResponseOk(msg=response)
        self.zero_send(req, rep)


def start_zero_engine(engine_args):
    from wde.microservices.standalone.server import Server
    assert "model" in engine_args

    from wde.microservices.framework.zero_manager.client import \
        ZeroManagerClient

    server = Server()
    server.setup()
    server.run(waiting=False)

    server_class = "wde.engine.zero_engine:ZeroEngine"

    manager_client = ZeroManagerClient(server.MANAGER_NAME)
    manager_client.wait_service_available(server.MANAGER_NAME)

    model_name = engine_args["model"]

    manager_client.start(name=model_name,
                         engine_kwargs={
                             "server_class": server_class,
                             "engine_args": engine_args
                         })
    return server
