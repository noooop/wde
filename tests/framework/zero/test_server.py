import time

import wde.envs as envs
from wde.microservices.framework.zero.server import ZeroServerProcess


def test_start_terminate():
    server_class = "wde.microservices.framework.zero.server:ZeroServer"

    h1 = ZeroServerProcess(server_class, {"do_register": False})
    h2 = ZeroServerProcess(server_class, {"do_register": False})
    h3 = ZeroServerProcess(server_class, {"do_register": False})

    h1.start()
    h2.start()
    h3.start()

    assert h1.is_alive()
    assert h2.is_alive()
    assert h3.is_alive()

    time.sleep(1)

    h1.terminate()
    h2.terminate()
    h3.terminate()

    assert not h1.is_alive()
    assert not h2.is_alive()
    assert not h3.is_alive()


def test_server_client():
    from wde.microservices.framework.zero.client import Z_Client
    from wde.microservices.framework.zero.schema import ZeroServerResponse

    server_class = "wde.microservices.framework.zero.server:Z_MethodZeroServer"
    h = ZeroServerProcess(server_class, {
        "do_register": False,
        "port": envs.NAME_SERVER_PORT
    })
    h.start()

    assert h.is_alive()

    try:
        client = Z_Client(f"tcp://localhost:{envs.NAME_SERVER_PORT}")

        assert client.query({"no_method": ""}) == ZeroServerResponse(
            **{
                'state':
                'error',
                'msg': [{
                    'type': 'missing',
                    'loc': 'method',
                    'msg': 'Field required',
                    'input': {
                        'no_method': ''
                    }
                }]
            })

        assert client.support_methods() == ZeroServerResponse(
            **{
                'state': 'ok',
                'msg': {
                    'name': 'Z_MethodZeroServer',
                    'support_methods': ['info', 'support_methods']
                }
            })

        data = {"method": "method_not_supported"}

        assert client.query(data) == ZeroServerResponse(
            **{
                'state': 'error',
                'msg': 'method [method_not_supported] not supported.'
            })
    finally:
        h.terminate()
        assert not h.is_alive()
