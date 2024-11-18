import inspect
from os import getpid
from queue import Queue

import gevent
import zmq
import zmq.asyncio

from wde.microservices.framework.zero.schema import (Timeout, ZeroMSQ,
                                                     ZeroServerResponse)
from wde.utils import random_uuid


class AsyncSocket(object):

    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.DEALER)
        self.socket.connect(addr)

    async def send(self, data):
        await self.socket.send(data, copy=False)

    async def send_multipart(self, data):
        await self.socket.send_multipart(data, copy=False)

    async def recv(self):
        return await self.socket.recv()

    async def recv_multipart(self):
        return await self.socket.recv_multipart()

    def close(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()

    def getsockopt(self, opt):
        return self.socket.getsockopt(opt)


class AsyncSocketPool(object):

    def __init__(self):
        self.queue = {}
        self.context = zmq.asyncio.Context()
        self._pid = getpid()

    def reinit(self):
        self.queue = {}
        self.context = zmq.asyncio.Context()
        self._pid = getpid()

    def get(self, addr):
        if self._pid != getpid():
            self.reinit()

        if addr not in self.queue:
            self.queue[addr] = Queue()

        queue = self.queue[addr]

        if queue.empty():
            return AsyncSocket(self.context, addr)
        else:
            return queue.get()

    def put(self, socket):
        if self._pid != getpid():
            self.reinit()
        else:
            addr = socket.addr
            if addr not in self.queue:
                self.queue[addr] = Queue()
            queue = self.queue[addr]
            queue.put(socket)

    def delete(self, socket):
        if self._pid != getpid():
            self.reinit()
        else:
            socket.close()


socket_pool = AsyncSocketPool()


class AsyncClient(object):
    timeout = 100000

    def __init__(self, addr):
        self.addr = addr

    async def _query(self, req, req_payload, n_try=3, timeout=None):
        _timeout = timeout or getattr(self, "timeout", None)

        for i in range(n_try):
            req_id = random_uuid().encode("utf-8")
            socket = socket_pool.get(self.addr)

            try:
                with gevent.Timeout(_timeout):
                    await socket.send_multipart([req_id, req] + req_payload)
                    out = await socket.recv_multipart()
            except gevent.timeout.Timeout:
                socket_pool.delete(socket)
                continue

            rep_id, msg, *payload = out

            if len(rep_id) == 22:
                socket_pool.put(socket)
                return out
            else:

                async def generator(out):
                    yield out

                    rep_id = out[0]
                    rcv_more = rep_id[22:23]

                    while rcv_more == b"M":
                        try:
                            with gevent.Timeout(_timeout):
                                out = await socket.recv_multipart()
                                rep_id = out[0]
                                rcv_more = rep_id[22:23]
                                yield out
                        except gevent.timeout.Timeout:
                            socket_pool.delete(socket)
                            raise Timeout(f"{self.addr} timeout")
                    socket_pool.put(socket)

                return generator(out)
        raise Timeout(f"{self.addr} timeout")

    async def query(self, data, **kwargs):
        req, req_payload = ZeroMSQ.load(data)
        response = await self._query(req, req_payload, **kwargs)
        if not inspect.isasyncgen(response):
            req_id, msg, *payload = response
            return ZeroServerResponse(**ZeroMSQ.unload(msg, payload))
        else:

            async def generator():
                async for req_id, msg, *payload in response:
                    yield ZeroServerResponse(**ZeroMSQ.unload(msg, payload))

            return generator()


class AsyncZ_Client(AsyncClient):

    async def support_methods(self, **kwargs):
        data = {"method": "support_methods"}
        return await self.query(data, **kwargs)
