import multiprocessing as mp
import time

import pytest

from tests.framework.core.util import (client_url, dummy_server, fake_server,
                                       server_url, stream_server)
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

    async def async_enumerate(aiterable, start=0):
        index = start
        async for value in aiterable:
            yield index, value
            index += 1

    try:
        for echo in range(1, 10):
            response = await client.query("stream", data={"echo": echo})
            async for i, rep in async_enumerate(response):
                assert i < echo
                assert rep.state == "ok"
                assert rep.msg == {'index': i}
    finally:
        s.terminate()
