import time
from random import choice

from cachetools import TTLCache, cached

from wde import envs
from wde.microservices.framework.nameserver.schema import (
    GetServiceNamesRequest, GetServicesRequest, ServerInfo)
from wde.microservices.framework.zero.client import Timeout, Z_Client

CLIENT_VALIDATION = True


class NameServerClient(Z_Client):
    timeout = 0.1

    def __init__(self, port=None):
        if port is None:
            self.port = envs.NAME_SERVER_PORT
        else:
            self.port = port
        Z_Client.__init__(self, f"tcp://localhost:{self.port}")

    def register(self, server_info):
        if CLIENT_VALIDATION:
            server_info = ServerInfo(**server_info).model_dump()

        data = {"method": "register", "data": server_info}
        return self.query(data)

    def deregister(self, server_info):
        if CLIENT_VALIDATION:
            server_info = ServerInfo(**server_info).model_dump()

        data = {"method": "deregister", "data": server_info}
        return self.query(data)

    def get_services(self, protocol, name):
        data = {"protocol": protocol, "name": name}

        if CLIENT_VALIDATION:
            data = GetServicesRequest(**data).model_dump()

        data = {"method": "get_services", "data": data}

        rep = self.query(data)
        if rep.state == "ok":
            rep.msg["services"] = [
                ServerInfo(**x) for x in rep.msg["services"]
            ]
        return rep

    def get_service_names(self, protocol):
        data = {"protocol": protocol}

        if CLIENT_VALIDATION:
            data = GetServiceNamesRequest(**data).model_dump()

        data = {"method": "get_service_names", "data": data}
        return self.query(data)


default_nameserver_client = NameServerClient()


class ZeroClient(object):

    def __init__(self, protocol, nameserver_port=None):
        self.protocol = protocol
        if nameserver_port is not None:
            self.nameserver_client = NameServerClient(nameserver_port)
        else:
            self.nameserver_client = default_nameserver_client

    def get_service_names(self):
        return self.nameserver_client.get_service_names(self.protocol)

    def get_services(self, name):
        rep = self.nameserver_client.get_services(self.protocol, name)

        if rep.state == "error":
            return None

        services = rep.msg["services"]
        if not services:
            return None

        return services

    @cached(cache=TTLCache(maxsize=1024, ttl=60))
    def get_services_cached(self, name):
        return self.get_services(name)

    def info(self, name):
        method = "info"
        client = self.get_client(name)
        if client is None:
            return None

        return self.query(name, method)

    def wait_service_available(self, name, timeout=10000):
        t = timeout + time.time()

        while time.time() < t:
            time.sleep(0.1)
            try:
                services = self.get_services(name)
                if services:
                    return services
            except Timeout:
                pass

        raise Timeout

    def get_client(self, name):
        services = self.get_services_cached(name)
        if not services:
            return None

        server = choice(services)
        client = Z_Client(f"tcp://{server.host}:{server.port}")
        return client

    def support_methods(self, name):
        method = "support_methods"
        client = self.get_client(name)
        if client is None:
            return None

        return self.query(name, method)

    def query(self, name, method, data=None, **kwargs):
        client = self.get_client(name)
        if client is None:
            return None
        if data is None:
            data = {}

        _data = {"method": method, "data": data}
        return client.query(_data, **kwargs)
