import inspect
from os import getpid
from queue import Queue

import gevent
import zmq.green as zmq

from wde.microservices.framework.zero.schema import (Timeout, ZeroMSQ,
                                                     ZeroServerResponse)
from wde.utils import random_uuid


class Socket(object):

    def __init__(self, context, addr):
        self.addr = addr
        self.socket = context.socket(zmq.DEALER)
        self.socket.connect(addr)

    def send(self, data):
        self.socket.send(data, copy=False)

    def send_multipart(self, data):
        self.socket.send_multipart(data, copy=False)

    def recv(self):
        return self.socket.recv(copy=False)

    def recv_multipart(self):
        return self.socket.recv_multipart(copy=False)

    def close(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()

    def getsockopt(self, opt):
        return self.socket.getsockopt(opt)


class SocketPool(object):

    def __init__(self):
        self.queue = {}
        self.context = zmq.Context.instance()
        self._pid = getpid()

    def reinit(self):
        self.queue = {}
        self.context = zmq.Context.instance()
        self._pid = getpid()

    def get(self, addr):
        if self._pid != getpid():
            self.reinit()

        if addr not in self.queue:
            self.queue[addr] = Queue()

        queue = self.queue[addr]

        if queue.empty():
            return Socket(self.context, addr)
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


socket_pool = SocketPool()


class Client(object):
    timeout = 100000

    def __init__(self, addr):
        self.addr = addr

    def _query(self, req, req_payload, n_try=3, timeout=None):
        _timeout = timeout or getattr(self, "timeout", None)

        for i in range(n_try):
            req_id = random_uuid().encode("utf-8")
            socket = socket_pool.get(self.addr)

            try:
                with gevent.Timeout(_timeout):
                    socket.send_multipart([req_id, req] + req_payload)
                    out = socket.recv_multipart()
            except gevent.timeout.Timeout:
                socket_pool.delete(socket)
                continue

            rep_id, msg, *payload = out
            rep_id = rep_id.bytes
            msg = msg.bytes

            out = [rep_id, msg] + payload

            if len(rep_id) == 22:
                socket_pool.put(socket)
                return out
            else:

                def generator(out):
                    yield out

                    rep_id = out[0]
                    rcv_more = rep_id[22:23]

                    while rcv_more == b"M":
                        try:
                            with gevent.Timeout(_timeout):
                                out = socket.recv_multipart()
                                rep_id, msg, *payload = out

                                rep_id = rep_id.bytes
                                msg = msg.bytes

                                out = [rep_id, msg] + payload

                                rcv_more = rep_id[22:23]
                                yield out
                        except gevent.timeout.Timeout:
                            socket_pool.delete(socket)
                            raise Timeout(f"{self.addr} timeout")
                    socket_pool.put(socket)

                return generator(out)
        raise Timeout(f"{self.addr} timeout")

    def query(self, data, **kwargs):
        req, req_payload = ZeroMSQ.load(data)
        response = self._query(req, req_payload, **kwargs)
        if not inspect.isgenerator(response):
            req_id, msg, *payload = response
            return ZeroServerResponse(**ZeroMSQ.unload(msg, payload))
        else:

            def generator():
                for req_id, msg, *payload in response:
                    yield ZeroServerResponse(**ZeroMSQ.unload(msg, payload))

            return generator()


class Z_Client(Client):

    def support_methods(self, **kwargs):
        data = {"method": "support_methods"}
        return self.query(data, **kwargs)
