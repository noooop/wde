import time

from wde.microservices.framework.nameserver.client import Timeout, ZeroClient
from wde.microservices.framework.zero_manager.schema import (StartRequest,
                                                             StatusRequest,
                                                             TerminateRequest)

CLIENT_VALIDATION = True


class ZeroManagerClient(ZeroClient):
    protocol = "manager"

    def __init__(self, name):
        self.name = name
        ZeroClient.__init__(self, self.protocol)

    def start(self, name, engine_kwargs=None):
        data = {"name": name, "engine_kwargs": engine_kwargs or {}}
        method = "start"

        if CLIENT_VALIDATION:
            data = StartRequest(**data).model_dump()

        return self.query(self.name, method, data)

    def terminate(self, name):
        method = "terminate"
        data = {
            "name": name,
        }
        if CLIENT_VALIDATION:
            data = TerminateRequest(**data).model_dump()
        return self.query(self.name, method, data)

    def list(self):
        method = "list"
        return self.query(self.name, method)

    def statuses(self):
        method = "statuses"
        return self.query(self.name, method)

    def status(self, name):
        method = "status"
        data = {
            "name": name,
        }
        if CLIENT_VALIDATION:
            data = StatusRequest(**data).model_dump()
        return self.query(self.name, method, data)

    def wait_service_status(self, name, timeout=10000, verbose=True):
        t = timeout + time.time()

        while time.time() < t:
            time.sleep(0.5)
            rep = self.status(name)
            if rep.state == "error":
                continue

            status = rep.msg["status"]
            exception = rep.msg["exception"]
            if status in ["prepare", "started"]:
                if verbose:
                    print(f"{name} {status}.")
            elif status in ["error"]:
                if verbose:
                    print(f"{name} {status}. {exception}.")
                return status, exception
            elif status in ["running"]:
                if verbose:
                    print(f"{name} available now.")
                return status, exception

        raise Timeout
