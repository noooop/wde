import inspect

from wde.microservices.framework.nameserver.async_client import AsyncZeroClient
from wde.microservices.framework.nameserver.client import ZeroClient
from wde.tasks.chat.schema.api import (PROTOCOL, ChatCompletionRequest,
                                       ChatCompletionResponse,
                                       ChatCompletionStreamResponse,
                                       ChatCompletionStreamResponseDone)

CLIENT_VALIDATION = True


class ChatClient(ZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        ZeroClient.__init__(self, self.protocol, nameserver_port)

    def chat(self, name, messages, tools=None, stream=False, options=None):
        method = "generate"
        data = {
            "model": name,
            "messages": messages,
            "tools": tools,
            "options": options or dict(),
            "stream": stream
        }
        if CLIENT_VALIDATION:
            data = ChatCompletionRequest(**data).dict()

        response = self.query(name, method, data)
        if response is None:
            raise RuntimeError(f"Chat [{name}] server not found.")

        if not inspect.isgenerator(response):
            if response.state != "ok":
                raise RuntimeError(
                    f"Chat [{name}] error, with error msg [{response.msg}]")

            rep = ChatCompletionResponse(**response.msg)
            return rep
        else:

            def generator():
                for rep in response:
                    if rep is None:
                        raise RuntimeError(f"Chat [{name}] server not found.")

                    if rep.state != "ok":
                        raise RuntimeError(
                            f"Chat [{name}] error, with error msg [{rep.msg}]")

                    if rep.msg["finish_reason"] is None:
                        rep = ChatCompletionStreamResponse(**rep.msg)
                    else:
                        rep = ChatCompletionStreamResponseDone(**rep.msg)

                    yield rep

            return generator()

    def stream_chat(self, name, messages, options=None):
        yield from self.chat(name, messages, stream=True, options=options)


class AsyncChatClient(AsyncZeroClient):
    protocol = PROTOCOL

    def __init__(self, nameserver_port=None):
        AsyncZeroClient.__init__(self, self.protocol, nameserver_port)

    async def chat(self,
                   name,
                   messages,
                   tools=None,
                   stream=False,
                   options=None):
        method = "generate"
        data = {
            "model": name,
            "messages": messages,
            "tools": tools,
            "options": options or dict(),
            "stream": stream
        }
        if CLIENT_VALIDATION:
            data = ChatCompletionRequest(**data).dict()

        response = await self.query(name, method, data)
        if response is None:
            raise RuntimeError(f"Chat [{name}] server not found.")

        if not inspect.isasyncgen(response):
            if response.state != "ok":
                raise RuntimeError(
                    f"Chat [{name}] error, with error msg [{response.msg}]")

            rep = ChatCompletionResponse(**response.msg)
            return rep
        else:

            async def generator():
                async for rep in response:
                    if rep is None:
                        raise RuntimeError(f"Chat [{name}] server not found.")

                    if rep.state != "ok":
                        raise RuntimeError(
                            f"Chat [{name}] error, with error msg [{rep.msg}]")

                    if rep.msg["finish_reason"] is None:
                        rep = ChatCompletionStreamResponse(**rep.msg)
                    else:
                        rep = ChatCompletionStreamResponseDone(**rep.msg)

                    yield rep

            return generator()

    async def stream_chat(self, name, messages, options=None):
        async for x in self.chat(name, messages, stream=True, options=options):
            yield x
