import time
from random import choice

from asyncache import cached
from cachetools import TTLCache

from wde import envs
from wde.microservices.framework.nameserver.schema import (
    GetServiceNamesRequest, GetServicesRequest, ServerInfo)
from wde.microservices.framework.zero.async_client import (AsyncZ_Client,
                                                           Timeout)

CLIENT_VALIDATION = True


class AsyncNameServerClient(AsyncZ_Client):
    timeout = 0.1

    def __init__(self, port=None):
        if port is None:
            self.port = envs.NAME_SERVER_PORT
        else:
            self.port = port
        AsyncZ_Client.__init__(self, f"tcp://localhost:{self.port}")

    async def register(self, server_info):
        if CLIENT_VALIDATION:
            server_info = ServerInfo(**server_info).model_dump()

        data = {"method": "register", "data": server_info}
        return await self.query(data)

    async def deregister(self, server_info):
        if CLIENT_VALIDATION:
            server_info = ServerInfo(**server_info).model_dump()

        data = {"method": "deregister", "data": server_info}
        return await self.query(data)

    async def get_services(self, protocol, name):
        data = {"protocol": protocol, "name": name}

        if CLIENT_VALIDATION:
            data = GetServicesRequest(**data).model_dump()

        data = {"method": "get_services", "data": data}

        rep = await self.query(data)
        if rep.state == "ok":
            rep.msg["services"] = [
                ServerInfo(**x) for x in rep.msg["services"]
            ]
        return rep

    async def get_service_names(self, protocol):
        data = {"protocol": protocol}

        if CLIENT_VALIDATION:
            data = GetServiceNamesRequest(**data).model_dump()

        data = {"method": "get_service_names", "data": data}
        return await self.query(data)


default_async_nameserver_client = AsyncNameServerClient()


class AsyncZeroClient:

    def __init__(self, protocol, nameserver_port=None):
        self.protocol = protocol
        if nameserver_port is not None:
            self.nameserver_client = AsyncNameServerClient(nameserver_port)
        else:
            self.nameserver_client = default_async_nameserver_client

    async def get_service_names(self):
        return await self.nameserver_client.get_service_names(self.protocol)

    async def get_services(self, name):
        rep = await self.nameserver_client.get_services(self.protocol, name)

        if rep.state == "error":
            return None

        services = rep.msg["services"]
        if not services:
            return None

        return services

    @cached(cache=TTLCache(maxsize=1024, ttl=60))
    async def get_services_cached(self, name):
        return await self.get_services(name)

    async def info(self, name):
        method = "info"
        client = self.get_client(name)
        if client is None:
            return None

        return await self.query(name, method)

    async def wait_service_available(self, name, timeout=10000):
        t = timeout + time.time()

        while time.time() < t:
            time.sleep(0.1)
            try:
                services = await self.get_services(name)
                if services:
                    return services
            except Timeout:
                pass

        raise Timeout

    async def get_client(self, name):
        services = await self.get_services_cached(name)
        if not services:
            return None

        server = choice(services)
        client = AsyncZ_Client(f"tcp://{server.host}:{server.port}")
        return client

    async def support_methods(self, name):
        method = "support_methods"
        client = self.get_client(name)
        if client is None:
            return None

        return await self.query(name, method)

    async def query(self, name, method, data=None, **kwargs):
        client = await self.get_client(name)
        if client is None:
            return None
        if data is None:
            data = {}

        _data = {"method": method, "data": data}
        return await client.query(_data, **kwargs)

    def check_response(self, name, rep, rep_cls):
        if rep is None:
            raise RuntimeError(
                f"{self.__class__.__name__} [{name}] server not found.")

        if rep.state == "ok":
            rep = rep_cls(**rep.msg)
        else:
            raise RuntimeError(
                f"{self.__class__.__name__} [{name}] error, with error msg [{rep.msg}]"
            )
        return rep
