import time

from wde import envs
from wde.client import ZeroManagerClient
from wde.microservices.framework.nameserver.client import ZeroClient
from wde.microservices.framework.zero.schema import ZeroServerResponseOk
from wde.microservices.framework.zero.server import ZeroServerProcess


def to_model_dump(msg):
    return ZeroServerResponseOk(msg=msg).model_dump()


def test_server():
    MANAGER_NAME = envs.ROOT_MANAGER_NAME
    server_class = "wde.microservices.framework.zero.server:Z_MethodZeroServer"
    test_protocols = ["protocol1", "protocol2"]
    n_server = 4

    nameserver = ZeroServerProcess(
        "wde.microservices.framework.nameserver.server:ZeroNameServer")
    manager = ZeroServerProcess(
        "wde.microservices.framework.zero_manager.server:ZeroManager",
        server_kwargs={
            "name": MANAGER_NAME,
            "server_class":
            "wde.core.framework.zero_manager.server:ZeroManager"
        })

    nameserver.start()
    manager.start()

    assert nameserver.is_alive()
    assert manager.is_alive()

    try:

        manager_client = ZeroManagerClient(MANAGER_NAME)

        print("# Wait manager available")
        manager_client.wait_service_available(MANAGER_NAME)

        assert manager_client.get_service_names().model_dump(
        ) == to_model_dump({'service_names': [MANAGER_NAME]})

        # manager support methods")
        assert manager_client.support_methods(
            MANAGER_NAME).model_dump() == to_model_dump({
                'name':
                'ZeroManager',
                'support_methods': [
                    'info', 'list', 'start', 'status', 'statuses',
                    'support_methods', 'terminate'
                ]
            })

        print("# init")
        assert manager_client.list().model_dump() == to_model_dump([])
        assert manager_client.statuses().model_dump() == to_model_dump({})

        print("# start")
        running_server = []
        for protocol in test_protocols:
            for n in range(n_server):
                server_name = f"{protocol}-{n}"

                assert manager_client.start(name=server_name,
                                            engine_kwargs={
                                                "server_class": server_class,
                                                "protocol": protocol
                                            }).model_dump() == to_model_dump(
                                                {'already_started': False})
                running_server.append(server_name)

                assert manager_client.list().model_dump() == to_model_dump(
                    running_server)

                time.sleep(0.1)

                assert manager_client.statuses().model_dump() == to_model_dump(
                    {name: 'running'
                     for name in running_server})

        print("# already_started")
        for protocol in test_protocols:
            for n in range(n_server):
                server_name = f"{protocol}-{n}"
                assert manager_client.start(
                    server_name).model_dump() == to_model_dump(
                        {'already_started': True})
                assert manager_client.list().model_dump() == to_model_dump(
                    running_server)
                assert manager_client.statuses().model_dump() == to_model_dump(
                    {name: 'running'
                     for name in running_server})

        print("# get_service_names")
        for protocol in test_protocols:
            client = ZeroClient(protocol)
            assert client.get_service_names().model_dump() == to_model_dump({
                'service_names': [f"{protocol}-{n}" for n in range(n_server)]
            })

        print("# link to server")
        for protocol in test_protocols:
            for n in range(n_server):
                server_name = f"{protocol}-{n}"
                client = ZeroClient(protocol)
                assert client.support_methods(
                    server_name).model_dump() == to_model_dump({
                        'name':
                        'Z_MethodZeroServer',
                        'support_methods': ['info', 'support_methods']
                    })

        print("# terminate")
        for protocol in test_protocols:
            for n in range(n_server):
                server_name = f"{protocol}-{n}"
                assert manager_client.terminate(
                    server_name).model_dump() == to_model_dump({
                        'founded':
                        True,
                        'last_status':
                        'running',
                        'last_exception':
                        None
                    })
                running_server.remove(server_name)
                assert manager_client.list().model_dump() == to_model_dump(
                    running_server)

        print("# not founded")
        for protocol in test_protocols:
            for n in range(n_server):
                server_name = f"{protocol}-{n}"
                assert manager_client.terminate(
                    server_name).model_dump() == to_model_dump(
                        {'founded': False})
                assert manager_client.list().model_dump() == to_model_dump([])

    finally:
        nameserver.terminate()
        manager.terminate()

        assert not nameserver.is_alive()
        assert not manager.is_alive()
