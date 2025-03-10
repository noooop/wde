import multiprocessing as mp
import time

import pytest

from tests.framework.core.util import (client_url, dummy_server, fake_server,
                                       server_url, stream_server)
from wde.microservices.framework.core.schema import ZeroClientTimeOut
from wde.microservices.framework.core.use_gevent.client import Client


def test_timeout1():
    client = Client(client_url, timeout=1)

    with pytest.raises(ZeroClientTimeOut):
        client.query("hello")


def test_timeout2():
    client = Client(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=fake_server, args=(server_url, ))
    s.start()

    with pytest.raises(ZeroClientTimeOut):
        client.query("hello")

    s.terminate()
    time.sleep(1)


def test_work_properly():
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


def test_stream():
    client = Client(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=stream_server, args=(server_url, ))
    s.start()

    try:
        for echo in range(1, 10):
            response = client.query("stream", data={"echo": echo})
            for i, rep in enumerate(response):
                assert i < echo
                assert rep.state == "ok"
                assert rep.msg == {'index': i}
    finally:
        s.terminate()