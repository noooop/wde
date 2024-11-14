from wde import envs
from wde.microservices.framework.nameserver.schema import ServerInfo
from wde.microservices.framework.zero.schema import ZeroServerResponseOk
from wde.microservices.framework.zero.server import ZeroServerProcess


def test_server():
    import wde.microservices.framework.nameserver.client as nameserver_client
    nameserver_client.CLIENT_VALIDATION = False

    from wde.client import NameServerClient

    nameserver = ZeroServerProcess(
        "wde.microservices.framework.nameserver.server:ZeroNameServer")

    nameserver.start()

    assert nameserver.is_alive()

    client = NameServerClient()

    name = "Test"
    protocol = "test"
    server_info = {
        "host": "localhost",
        "port": envs.NAME_SERVER_PORT,
        "name": name,
        "protocol": protocol
    }
    try:
        # NameServer support_methods
        assert client.support_methods().model_dump() == ZeroServerResponseOk(
            msg={
                'name':
                'ZeroNameServer',
                'support_methods': [
                    'deregister', 'get_service_names', 'get_services', 'info',
                    'register', 'support_methods'
                ]
            }).model_dump()

        # init
        assert client.get_service_names(
            protocol).model_dump() == ZeroServerResponseOk(msg={
                'service_names': []
            }).model_dump()
        assert client.get_services(
            protocol, name).model_dump() == ZeroServerResponseOk(msg={
                'services': []
            }).model_dump()

        # register
        assert client.register(server_info).model_dump(
        ) == ZeroServerResponseOk(msg={
            'register': 'success'
        }).model_dump()

        assert client.get_service_names(
            protocol).model_dump() == ZeroServerResponseOk(msg={
                'service_names': ['Test']
            }).model_dump()
        assert client.get_services(
            protocol, name).model_dump() == ZeroServerResponseOk(msg={
                'services': [ServerInfo(**server_info)]
            }).model_dump()

        # deregister
        assert client.deregister(
            server_info).model_dump() == ZeroServerResponseOk(msg={
                'founded': True
            }).model_dump()
        assert client.get_service_names(
            protocol).model_dump() == ZeroServerResponseOk(msg={
                'service_names': []
            }).model_dump()
        assert client.get_services(
            protocol, name).model_dump() == ZeroServerResponseOk(msg={
                'services': []
            }).model_dump()

    finally:
        nameserver.terminate()
        assert not nameserver.is_alive()
