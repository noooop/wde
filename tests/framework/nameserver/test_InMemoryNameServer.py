from wde import envs
from wde.microservices.framework.nameserver.schema import ServerInfo
from wde.microservices.framework.nameserver.server import InMemoryNameServer


def test_InMemoryNameServer():
    nameserver = InMemoryNameServer()
    name = "Qwen/Qwen1.5-0.5B-Chat"
    protocol = "chat"
    server_info = ServerInfo(
        **{
            "host": "localhost",
            "port": envs.NAME_SERVER_PORT,
            "name": name,
            "protocol": protocol
        })

    assert nameserver.get_service_names(protocol) == []
    assert nameserver.get_services(protocol, name) == []

    # register
    nameserver.register(server_info)
    assert nameserver.get_service_names(protocol) == [name]
    assert nameserver.get_services(protocol, name) == [server_info]

    # deregister
    nameserver.deregister(server_info)
    assert nameserver.get_service_names(protocol) == []
    assert nameserver.get_services(protocol, name) == []
