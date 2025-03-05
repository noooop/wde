import multiprocessing as mp
import time

import pytest

from tests.framework.core.util import (client_url, dummy_server, fake_server,
                                       server_url)


@pytest.mark.asyncio
async def test_timeout1():
    from wde.microservices.framework.core.schema import ZeroClientTimeOut
    from wde.microservices.framework.core.use_asyncio.client import AsyncClient

    client = AsyncClient(client_url, timeout=1)

    with pytest.raises(ZeroClientTimeOut):
        await client.query("hello")


@pytest.mark.asyncio
async def test_timeout2():
    from wde.microservices.framework.core.schema import ZeroClientTimeOut
    from wde.microservices.framework.core.use_asyncio.client import AsyncClient

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
    from wde.microservices.framework.core.use_asyncio.client import AsyncClient

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
