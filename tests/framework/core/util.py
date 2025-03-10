port = "9527"

client_url = f"tcp://localhost:{port}"
server_url = f"tcp://*:{port}"

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
