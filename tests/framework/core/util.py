port = "9527"

client_url = f"tcp://localhost:{port}"
server_url = f"tcp://*:{port}"


asyncio_default_fixture_loop_scope="function"

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