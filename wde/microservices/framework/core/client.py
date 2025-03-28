import sys

from wde.utils import lazy_import


class ZeroClient:
    naive_client_cls = "wde.microservices.framework.core.use_naive.client:Client"
    gevent_client_cls = "wde.microservices.framework.core.use_gevent.client:Client"
    asyncio_client_cls = "wde.microservices.framework.core.use_asyncio.client:AsyncClient"

    def __init__(self, addr: str, timeout=None):
        self.addr = addr

        if timeout is not None:
            self.timeout = timeout
        else:
            self.timeout = None

        self.naive_client = None
        self.gevent_client = None
        self.asyncio_client = None

    def ensure_naive_client(self):
        if self.naive_client is None:
            self.naive_client = lazy_import(self.naive_client_cls)(
                self.addr, self.timeout)

    def ensure_gevent_client(self):
        if self.gevent_client is None:
            self.gevent_client = lazy_import(self.gevent_client_cls)(
                self.addr, self.timeout)

    def ensure_asyncio_client(self):
        if self.asyncio_client is None:
            self.asyncio_client = lazy_import(self.asyncio_client_cls)(
                self.addr, self.timeout)

    def ensure_sync_client(self):
        if self.naive_client is not None or self.gevent_client is not None:
            return

        # need a better way to detect gevent
        if "gevent.event" in sys.modules:
            self.ensure_gevent_client()
        else:
            self.ensure_naive_client()

    def query(self, method_name, data=None, **kwargs):
        self.ensure_sync_client()

        client = self.gevent_client or self.naive_client

        return client.query(method_name, data, **kwargs)

    async def aquery(self, method_name, data=None, **kwargs):
        self.ensure_asyncio_client()

        client = self.asyncio_client

        return await client.query(method_name, data, **kwargs)
