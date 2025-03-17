import json
import traceback
from typing import Any, List, Optional, Tuple

from wde.logger import init_logger
from wde.microservices.framework.core.schema import (
    ValidationError, ZeroMSQ, ZeroServerRequest, ZeroServerResponse,
    ZeroServerResponseError, ZeroServerResponseOk, ZeroServerStreamResponseOk,
    convert_errors)
from wde.utils import lazy_import

logger = init_logger(__name__)


class ZeroEngine:

    def __init__(
        self,
        socket,
        port,
    ):
        self.socket = socket
        self.port = port

    def process(self,
                msg) -> Tuple[Optional[ZeroServerRequest], Optional[Any]]:
        try:
            client_id, task, metadata, *payload = msg
            client_id, task, metadata = client_id.bytes, task.bytes, metadata.bytes
            method_name, task_id = task.split(b"@")
            method_name = method_name.decode("utf-8")

        except Exception:
            traceback.print_exc()
            return None, None

        try:
            data = ZeroMSQ.unload(metadata, payload)
        except json.JSONDecodeError as e:
            self.handle_error(str(e), client_id, task_id)
            return None, None

        try:
            req = ZeroServerRequest(method_name=method_name,
                                    client_id=client_id,
                                    task_id=task_id)
        except ValidationError as e:
            self.handle_error(e, client_id, task_id)
            return None, None

        return req, data

    def handle_error(self, err_msg, client_id, task_id):
        if isinstance(err_msg, ValidationError):
            err_msg = convert_errors(err_msg)

        rep = ZeroServerResponseError(msg=err_msg)

        data, payload = ZeroMSQ.load(rep)
        self.socket.send_multipart([client_id, task_id, data] + payload)

    def zero_send(self, req: ZeroServerRequest, rep: ZeroServerResponse):
        if isinstance(rep, ZeroServerStreamResponseOk):
            rep_id = req.task_id + (b"M" if rep.snd_more else b"N") + str(
                rep.rep_id).encode("utf-8")
        else:
            rep_id = req.task_id

        data, payload = ZeroMSQ.load(rep)

        self.socket.send_multipart([req.client_id, rep_id, data] + payload)


class MetaZeroEngine(ZeroEngine):

    def __init__(self, engines: Optional[List[str]] = None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.engines = {"support_methods": self.support_methods}

        if engines is not None:
            for engine in engines:
                engine = lazy_import(engine)
                self.engines[engine.method_name] = engine(
                    self.socket, self.port)

    def default(self, req: ZeroServerRequest, data: Any):
        method_name = req.method_name
        err_msg = f"method [{method_name}] not supported."
        self.handle_error(err_msg,
                          client_id=req.client_id,
                          task_id=req.task_id)

    def support_methods(self, req: ZeroServerRequest, data: Any):
        _support_methods = ["support_methods"]
        rep = ZeroServerResponseOk(msg={
            "name": self.__class__.__name__,
            "support_methods": _support_methods
        })
        self.zero_send(req, rep)

    def __call__(self, msg):
        req, data = self.process(msg)
        if req is None:
            return

        engine = self.engines.get(req.method_name, self.default)
        engine(req, data)


class SyncZeroEngine(ZeroEngine):
    method_name = ""
    request_class = None

    def __call__(self, req: ZeroServerRequest, data: Any):
        if self.request_class is not None:
            try:
                data = self.request_class(**data)
            except ValidationError as e:
                self.handle_error(e, req.client_id, req.task_id)
                return None, None

        self.call(req, data)

    def call(self, req: ZeroServerRequest, data: Any):
        raise NotImplementedError
