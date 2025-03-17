import multiprocessing as mp
import time

import pytest

from tests.framework.core.util import (client_url, dummy_server, fake_server,
                                       redirect_port, redirect_server,
                                       server_port, server_url, stream_server)
from wde.microservices.framework.core.schema import ZeroClientTimeOut
from wde.microservices.framework.core.use_asyncio.client import AsyncClient


@pytest.mark.asyncio
async def test_timeout1():
    client = AsyncClient(client_url, timeout=1)

    with pytest.raises(ZeroClientTimeOut):
        await client.query("hello")


@pytest.mark.asyncio
async def test_timeout2():

    client = AsyncClient(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=fake_server, args=(server_url, ))
    s.start()

    try:
        with pytest.raises(ZeroClientTimeOut):
            await client.query("hello")
    finally:
        s.terminate()
        time.sleep(1)


@pytest.mark.asyncio
async def test_work_properly():

    client = AsyncClient(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=dummy_server, args=(server_url, ))
    s.start()

    try:
        respons = await client.query("hello")

        assert respons.state == "ok"
        assert respons.msg == "world!"
    finally:
        s.terminate()


@pytest.mark.asyncio
async def test_stream():
    client = AsyncClient(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=stream_server, args=(server_url, ))
    s.start()

    try:
        for echo in range(1, 10):
            response = await client.query("stream", data={"echo": echo})
            index = 0
            async for rep in response:
                assert rep.state == "ok"
                assert rep.msg == {'index': index}
                index += 1

            assert index == echo
    finally:
        s.terminate()


@pytest.mark.asyncio
async def test_redirect():
    client = AsyncClient(client_url, timeout=1)

    ctx = mp.get_context('spawn')
    s = ctx.Process(target=redirect_server, args=(server_port, redirect_port))
    s.start()

    try:
        for echo in range(1, 10):
            response = await client.query("stream", data={"echo": echo})

            assert response.state == 'redirect'
            assert response.protocol == "zmq"

            redirect_client = AsyncClient(response.url,
                                          timeout=response.timeout)
            redirect_response = await redirect_client.query(response.access_key
                                                            )

            index = 0
            async for rep in redirect_response:
                assert rep.state == "ok"
                assert rep.msg == {'index': index}
                index += 1

            assert index == echo
    finally:
        s.terminate()
