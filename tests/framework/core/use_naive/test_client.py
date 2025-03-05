import multiprocessing as mp
import time

import pytest

from tests.framework.core.util import (client_url, dummy_server, fake_server,
                                       server_url)


def test_timeout1():
    from wde.microservices.framework.core.schema import ZeroClientTimeOut
    from wde.microservices.framework.core.use_naive.client import Client

    client = Client(client_url, timeout=1)

    with pytest.raises(ZeroClientTimeOut):
        client.query("hello")


def test_timeout2():
    from wde.microservices.framework.core.schema import ZeroClientTimeOut
    from wde.microservices.framework.core.use_naive.client import Client

    client = Client(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=fake_server, args=(server_url, ))
    s.start()

    with pytest.raises(ZeroClientTimeOut):
        client.query("hello")

    s.terminate()
    time.sleep(1)


def test_work_properly():
    from wde.microservices.framework.core.use_naive.client import Client
    client = Client(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=dummy_server, args=(server_url, ))
    s.start()

    try:

        respons = client.query("hello")

        assert respons.state == "ok"
        assert respons.msg == "world!"
    finally:
        s.terminate()
