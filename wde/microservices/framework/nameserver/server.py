from wde.microservices.framework.nameserver.schema import (
    GetServiceNamesRequest, GetServicesRequest, ServerInfo)
from wde.microservices.framework.zero.schema import (ZeroServerRequest,
                                                     ZeroServerResponseOk)
from wde.microservices.framework.zero.server import Z_MethodZeroServer

NameServerPort = 9527


class NameServerInterfaces(object):

    def register(self, server_info: ServerInfo):
        raise NotImplementedError

    def deregister(self, server_info: ServerInfo):
        # https://english.stackexchange.com/questions/25931/unregister-vs-deregister
        raise NotImplementedError

    def get_services(self, protocol, name):
        raise NotImplementedError

    def get_service_names(self, protocol):
        raise NotImplementedError


class InMemoryNameServer(NameServerInterfaces):

    def __init__(self):
        self.domain = dict()

    def register(self, server_info: ServerInfo):
        name = server_info.name
        protocol = server_info.protocol

        if protocol not in self.domain:
            self.domain[protocol] = {}

        if name not in self.domain[protocol]:
            self.domain[protocol][name] = []

        self.domain[protocol][name].append(server_info)

    def deregister(self, server_info: ServerInfo):
        name = server_info.name
        protocol = server_info.protocol

        if protocol not in self.domain:
            return False

        if name not in self.domain[protocol]:
            return False

        services = self.domain[protocol][name]

        if server_info not in services:
            return False

        services = [s for s in services if s != server_info]
        if len(services) > 0:
            self.domain[protocol][name] = services
        else:
            del self.domain[protocol][name]

        return True

    def get_services(self, protocol, name):
        services = self.domain.get(protocol, dict()).get(name, list())
        return services

    def get_service_names(self, protocol):
        service_names = list(self.domain.get(protocol, dict()).keys())
        return service_names


class ZeroNameServer(Z_MethodZeroServer):

    def __init__(self, port=None, nameserver_class=None, **kwargs):
        Z_MethodZeroServer.__init__(self,
                                    port=port or NameServerPort,
                                    do_register=False,
                                    **kwargs)

        if nameserver_class is None:
            self.nameserver_class = InMemoryNameServer
        else:
            self.nameserver_class = nameserver_class

        self._nameserver = None

    def init(self):
        self._nameserver = self.nameserver_class()
        print(f"ZeroNameServer: {self.nameserver_class.__name__} running!",
              "port:", self.port)

    def z_register(self, req: ZeroServerRequest):
        server_info = ServerInfo(**req.data)
        self._nameserver.register(server_info)

        rep = ZeroServerResponseOk(msg={"register": "success"})
        self.zero_send(req, rep)

    def z_deregister(self, req):
        server_info = ServerInfo(**req.data)
        founded = self._nameserver.deregister(server_info)
        rep = ZeroServerResponseOk(msg={"founded": founded})
        self.zero_send(req, rep)

    def z_get_services(self, req):
        kwargs = GetServicesRequest(**req.data)
        services = self._nameserver.get_services(kwargs.protocol, kwargs.name)

        rep = ZeroServerResponseOk(msg={"services": services})
        self.zero_send(req, rep)

    def z_get_service_names(self, req):
        kwargs = GetServiceNamesRequest(**req.data)
        service_names = self._nameserver.get_service_names(kwargs.protocol)
        rep = ZeroServerResponseOk(msg={"service_names": service_names})
        self.zero_send(req, rep)


def nameserver(nameserver_class=None):
    nameserver_class = nameserver_class or InMemoryNameServer
    server = ZeroNameServer(nameserver_class=nameserver_class)
    server.start()


if __name__ == '__main__':
    nameserver()
