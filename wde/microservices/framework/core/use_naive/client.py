from os import getpid
from queue import Queue

import zmq

from wde.microservices.framework.core.interface import ClientInterface
from wde.microservices.framework.core.schema import ZeroClientTimeOut
from wde.utils import random_uuid


class Socket:

    def __init__(self, context: zmq.Context, addr: str):
        self.addr = addr
        self.socket = context.socket(zmq.DEALER)
        self.socket.connect(addr)

    def set_timeout(self, timeout):
        if timeout is None:
            timeout = -1
        else:
            timeout = int(timeout * 1000)

        self.socket.RCVTIMEO = timeout
        self.socket.SNDTIMEO = timeout

    def send_multipart(self, data):
        self.socket.send_multipart(data, copy=False)

    def recv_multipart(self):
        return self.socket.recv_multipart(copy=False)

    def close(self):
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.close()


class SocketPool:

    def __init__(self, addr: str):
        self.addr = addr

        self.queue: Queue
        self.context: zmq.Context
        self._pid: int

        self.reinit()

    def reinit(self):
        self.queue = Queue()
        self.context = zmq.Context.instance()
        self._pid = getpid()

    def get(self):
        if self._pid != getpid():
            self.reinit()

        if self.queue.empty():
            return Socket(context=self.context, addr=self.addr)
        else:
            return self.queue.get()

    def put(self, socket):
        if self._pid != getpid():
            self.reinit()
        else:
            self.queue.put(socket)

    def close(self, socket):
        if self._pid != getpid():
            self.reinit()
        else:
            socket.close()


class Client(ClientInterface):

    def __init__(self, addr: str, timeout=None):
        if timeout is not None:
            self.timeout = timeout
        else:
            self.timeout = None

        self.socket_pool = SocketPool(addr=addr)

    @property
    def addr(self):
        return self.socket_pool.addr

    def _query(self, method_name, metadata, payload, n_try=3, timeout=None):
        _timeout = timeout or getattr(self, "timeout", None)

        for i in range(n_try):
            socket = self.socket_pool.get()
            socket.set_timeout(_timeout)

            task_id = random_uuid()

            task = [method_name, task_id]
            task = b"@".join([s.encode("utf-8") for s in task])
            try:
                socket.send_multipart([task, metadata] + payload)
                response = socket.recv_multipart()
            except zmq.error.Again:
                self.socket_pool.close(socket)
                continue

            rep_id, msg, *payload = response
            rep_id = rep_id.bytes
            msg = msg.bytes

            response = [rep_id, msg] + payload

            if len(rep_id) == 22:
                self.socket_pool.put(socket)
                return response
            else:

                def generator(response):
                    yield response

                    rep_id = response[0]
                    rcv_more = rep_id[22:23]
                    payload = response[2:]

                    while rcv_more == b"M":
                        try:
                            socket.send_multipart([task, metadata] + payload)
                            response = socket.recv_multipart()
                        except zmq.error.Again:
                            self.socket_pool.close(socket)
                            raise ZeroClientTimeOut(f"{self.addr} timeout.")

                        rep_id, msg, *payload = response
                        rep_id = rep_id.bytes
                        msg = msg.bytes

                        response = [rep_id, msg] + payload
                        rcv_more = rep_id[22:23]

                        yield response

                    self.socket_pool.put(socket)

                return generator(response)
        else:
            raise ZeroClientTimeOut(f"{self.addr} timeout.")
