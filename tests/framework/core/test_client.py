import time

import pytest

from tests.framework.core.util import client_url, server_port
from wde.microservices.framework.core.client import ZeroClient
from wde.microservices.framework.core.use_naive.server import ZeroServerProcess


def start_dummy_engine():
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
    return s


def test_naive():

    s = start_dummy_engine()

    client = ZeroClient(client_url, timeout=1)

    try:

        response = client.query("hello")

        assert response.state == "ok"
        assert response.msg == "world!"
    finally:
        s.terminate()
        assert not s.is_alive()

    assert client.naive_client is not None
    assert client.gevent_client is None
    assert client.asyncio_client is None

    time.sleep(1)


def test_gevent():
    from gevent.pool import Pool

    s = start_dummy_engine()

    client = ZeroClient(client_url, timeout=1)

    def worker(x):
        response = client.query("hello")

        assert response.state == "ok"
        assert response.msg == "world!"

    try:
        p = Pool(10)

        for i in p.imap_unordered(worker, range(100)):
            pass

    finally:
        s.terminate()
        assert not s.is_alive()

    assert client.naive_client is None
    assert client.gevent_client is not None
    assert client.asyncio_client is None

    time.sleep(1)


@pytest.mark.asyncio
async def test_asyncio():

    s = start_dummy_engine()

    client = ZeroClient(client_url, timeout=1)

    try:
        respons = await client.aquery("hello")

        assert respons.state == "ok"
        assert respons.msg == "world!"
    finally:
        s.terminate()
        assert not s.is_alive()

    assert client.naive_client is None
    assert client.gevent_client is None
    assert client.asyncio_client is not None
