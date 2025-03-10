import inspect

from wde.microservices.framework.core.schema import ZeroMSQ, ZeroServerResponse


class ClientInterface:

    @property
    def addr(self):
        raise NotImplementedError

    def _query(self, method_name, metadata, payload, n_try=3, timeout=None):
        raise NotImplementedError

    def query(self, method_name, data=None, **kwargs):
        if data is None:
            data = {}

        metadata, payload = ZeroMSQ.load(data)
        response = self._query(method_name, metadata, payload, **kwargs)

        if not inspect.isgenerator(response):
            task_id, metadata, *payload = response
            return ZeroServerResponse(**ZeroMSQ.unload(metadata, payload))
        else:

            def generator():
                for req_id, msg, *payload in response:
                    yield ZeroServerResponse(**ZeroMSQ.unload(msg, payload))

            return generator()


class AsyncClientInterface:

    @property
    def addr(self):
        raise NotImplementedError

    async def _query(self,
                     method_name,
                     metadata,
                     payload,
                     n_try=3,
                     timeout=None):
        raise NotImplementedError

    async def query(self, method_name, data=None, **kwargs):
        if data is None:
            data = {}

        metadata, payload = ZeroMSQ.load(data)
        response = await self._query(method_name, metadata, payload, **kwargs)

        if not inspect.isasyncgen(response):
            task_id, metadata, *payload = response
            return ZeroServerResponse(**ZeroMSQ.unload(metadata, payload))
        else:

            async def generator():
                async for req_id, msg, *payload in response:
                    yield ZeroServerResponse(**ZeroMSQ.unload(msg, payload))

            return generator()
