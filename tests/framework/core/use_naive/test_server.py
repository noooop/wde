import time

from tests.framework.core.util import server_port
from wde.microservices.framework.core.schema import ZeroServerResponse
from wde.microservices.framework.core.use_naive.client import Client
from wde.microservices.framework.core.use_naive.server import ZeroServerProcess


def test_start_terminate():
    engine_class = "wde.microservices.framework.core.engine:MetaZeroEngine"

    s1 = ZeroServerProcess(engine_class, server_kwargs={"do_register": False})
    s2 = ZeroServerProcess(engine_class, server_kwargs={"do_register": False})
    s3 = ZeroServerProcess(engine_class, server_kwargs={"do_register": False})

    assert s1.status == "prepare"
    assert s2.status == "prepare"
    assert s3.status == "prepare"

    s1.start()
    s2.start()
    s3.start()

    time.sleep(1)

    assert s1.is_alive()
    assert s2.is_alive()
    assert s3.is_alive()

    assert s1.status == "running"
    assert s2.status == "running"
    assert s3.status == "running"

    s1.terminate()
    s2.terminate()
    s3.terminate()

    time.sleep(1)

    assert not s1.is_alive()
    assert not s2.is_alive()
    assert not s3.is_alive()

    assert s1.status == "stopped"
    assert s2.status == "stopped"
    assert s3.status == "stopped"


def test_supported_method():

    engine_class = "wde.microservices.framework.core.engine:MetaZeroEngine"
    s = ZeroServerProcess(engine_class,
                          server_kwargs={
                              "do_register": False,
                              "port": server_port
                          })
    s.start()

    assert s.is_alive()

    try:
        client = Client(f"tcp://localhost:{server_port}", timeout=1.)

        assert (client.query(
            method_name="not_supported_method") == ZeroServerResponse(
                state='error',
                msg='method [not_supported_method] not supported.'))

        assert (client.query(
            method_name="support_methods") == ZeroServerResponse(
                state='ok',
                msg={
                    'name': 'MetaZeroEngine',
                    'support_methods': ['support_methods']
                }))

    finally:
        s.terminate()
        assert not s.is_alive()


def test_work_properly():
    engine_class = "wde.microservices.framework.core.engine:MetaZeroEngine"
    dummy_engine = "tests.framework.core.util:DummyEngine"

    s = ZeroServerProcess(engine_class,
                          server_kwargs={
                              "do_register": False,
                              "port": server_port
                          },
                          engine_kwargs={"engines": [dummy_engine]})
    s.start()

    assert s.is_alive()

    client = Client(f"tcp://localhost:{server_port}", timeout=1)

    try:

        response = client.query("hello")

        assert response.state == "ok"
        assert response.msg == "world!"
    finally:
        s.terminate()

    time.sleep(1)


def test_stream():
    engine_class = "wde.microservices.framework.core.engine:MetaZeroEngine"
    stream_engine = "tests.framework.core.util:StreamEngine"

    s = ZeroServerProcess(engine_class,
                          server_kwargs={
                              "do_register": False,
                              "port": server_port
                          },
                          engine_kwargs={"engines": [stream_engine]})
    s.start()

    assert s.is_alive()

    client = Client(f"tcp://localhost:{server_port}", timeout=1)
    try:
        for echo in range(1, 10):
            response = client.query("stream", data={"echo": echo})
            index = 0
            for rep in response:
                assert rep.state == "ok"
                assert rep.msg == {'index': index}
                index += 1

            assert index == echo
    finally:
        s.terminate()