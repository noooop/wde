from wde.microservices.framework.core.schema import ZeroServerRedirect
from wde.utils import random_uuid

server_port = "9527"
redirect_port = "9528"

client_url = f"tcp://localhost:{server_port}"
server_url = f"tcp://*:{server_port}"

asyncio_default_fixture_loop_scope = "function"


def fake_server(url):
    import zmq
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.bind(url)

    while True:
        socket.recv_multipart(copy=False)
        # don't send anything


def dummy_server(url):
    import zmq

    from wde.microservices.framework.core.schema import (ZeroMSQ,
                                                         ZeroServerResponse)
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.bind(url)

    while True:
        client_id, task, metadata, *payload = socket.recv_multipart(copy=False)
        client_id, task, metadata = client_id.bytes, task.bytes, metadata.bytes
        method_name, task_id = task.split(b"@")

        assert method_name == b"hello"

        response = ZeroServerResponse(state="ok", msg="world!")
        data, payload = ZeroMSQ.load(response)

        socket.send_multipart([client_id, task_id, data] + payload)


def stream_server(url):
    import zmq

    from wde.microservices.framework.core.schema import (
        ZeroMSQ, ZeroServerRequest, ZeroServerStreamResponseOk)
    context = zmq.Context.instance()
    socket = context.socket(zmq.ROUTER)
    socket.bind(url)

    while True:
        client_id, task, metadata, *payload = socket.recv_multipart(copy=False)
        client_id, task, metadata = client_id.bytes, task.bytes, metadata.bytes
        method_name, task_id = task.split(b"@")

        assert method_name == b"stream"

        data = ZeroMSQ.unload(metadata, payload)

        req = ZeroServerRequest(method_name=method_name,
                                client_id=client_id,
                                task_id=task_id,
                                data=data)

        echo = req.data["echo"]

        for i in range(echo):
            snd_more = i < echo - 1

            rep = ZeroServerStreamResponseOk(msg={"index": i},
                                             snd_more=snd_more,
                                             rep_id=i)

            rep_id = req.task_id + (b"M" if rep.snd_more else b"N") + str(
                rep.rep_id).encode("utf-8")

            data, payload = ZeroMSQ.load(rep)

            socket.send_multipart([req.client_id, rep_id, data] + payload)


def redirect_server(server_port, redirect_port):
    import zmq

    from wde.microservices.framework.core.schema import (
        ZeroMSQ, ZeroServerRequest, ZeroServerStreamResponseOk)
    context = zmq.Context.instance()
    server_socket = context.socket(zmq.ROUTER)
    server_socket.bind(f"tcp://*:{server_port}")

    redirect_socket = context.socket(zmq.ROUTER)
    redirect_socket.bind(f"tcp://*:{redirect_port}")

    def server_part():
        client_id, task, metadata, *payload = server_socket.recv_multipart(
            copy=False)
        client_id, task, metadata = client_id.bytes, task.bytes, metadata.bytes
        method_name, task_id = task.split(b"@")

        assert method_name == b"stream"

        data = ZeroMSQ.unload(metadata, payload)

        req = ZeroServerRequest(method_name=method_name,
                                client_id=client_id,
                                task_id=task_id,
                                data=data)

        echo = req.data["echo"]

        assert echo > 0

        access_key = random_uuid()

        rep = ZeroServerRedirect(url=f"tcp://localhost:{redirect_port}",
                                 access_key=access_key,
                                 timeout=2.)

        data, payload = ZeroMSQ.load(rep)

        server_socket.send_multipart([req.client_id, req.task_id, data] +
                                     payload)

        return access_key, req

    def redirect_part(access_key, req):
        goon = True
        while goon:
            client_id, task, metadata, *payload = redirect_socket.recv_multipart(
                copy=False)
            client_id, task, metadata = client_id.bytes, task.bytes, metadata.bytes
            method_name, task_id = task.split(b"@")

            if method_name == access_key.encode("utf-8"):
                goon = False
            else:
                continue

            echo = req.data["echo"]

            for i in range(echo):
                snd_more = i < echo - 1

                rep = ZeroServerStreamResponseOk(msg={"index": i},
                                                 snd_more=snd_more,
                                                 rep_id=i)

                rep_id = req.task_id + (b"M" if rep.snd_more else b"N") + str(
                    rep.rep_id).encode("utf-8")

                data, payload = ZeroMSQ.load(rep)

                redirect_socket.send_multipart([client_id, rep_id, data] +
                                               payload)

    while True:
        access_key, req = server_part()
        redirect_part(access_key, req)
