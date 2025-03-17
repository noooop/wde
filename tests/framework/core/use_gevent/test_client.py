import multiprocessing as mp
import time

import pytest

from tests.framework.core.util import (client_url, dummy_server, fake_server,
                                       redirect_port, redirect_server,
                                       server_port, server_url, stream_server)
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
            index = 0
            for rep in response:
                assert rep.state == "ok"
                assert rep.msg == {'index': index}
                index += 1

            assert index == echo
    finally:
        s.terminate()


def test_redirect():
    client = Client(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=redirect_server, args=(server_port, redirect_port))
    s.start()

    try:
        for echo in range(1, 10):
            response = client.query("stream", data={"echo": echo})

            assert response.state == 'redirect'
            assert response.protocol == "zmq"

            redirect_client = Client(response.url, timeout=response.timeout)
            redirect_response = redirect_client.query(response.access_key)

            index = 0
            for rep in redirect_response:
                assert rep.state == "ok"
                assert rep.msg == {'index': index}
                index += 1

            assert index == echo
    finally:
        s.terminate()
