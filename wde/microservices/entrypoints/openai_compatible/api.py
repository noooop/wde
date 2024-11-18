import base64
import inspect
import time
import traceback
from typing import List, Literal, Union

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import wde.microservices.entrypoints.openai_compatible.schema as protocol
from wde.client import (AsyncChatClient, AsyncRerankerClient,
                        AsyncRetrieverClient)
from wde.tasks.chat.schema.api import ChatCompletionStreamResponseDone
from wde.utils import random_uuid

chat_client = AsyncChatClient()
retriever_client = AsyncRetrieverClient()
reranker_client = AsyncRerankerClient()
app = FastAPI()


@app.get("/")
async def health():
    return "wde is running"


@app.get("/v1/models")
async def models():
    response = await retriever_client.get_service_names()
    services = response.msg["service_names"]
    return protocol.ModelList(
        data=[protocol.ModelCard(id=s) for s in services])


@app.get("/v1/models/{model_id:path}", name="path-convertor")
async def model(model_id: str):
    return protocol.ModelCard(id=model_id)


def _get_embedding(
    embedding,
    encoding_format: Literal["float", "base64"],
) -> Union[List[float], str]:
    if encoding_format == "float":
        return embedding.tolist()
    elif encoding_format == "base64":
        embedding_bytes = np.array(embedding, dtype="float32").tobytes()
        return base64.b64encode(embedding_bytes).decode("utf-8")


@app.post("/v1/embeddings")
async def embeddings(req: protocol.EmbeddingRequest):
    try:
        rep = await retriever_client.encode(req.model, req.input)

        data = []
        for idx, res in enumerate([rep]):

            embedding = _get_embedding(res.embedding, req.encoding_format)
            embedding_data = protocol.EmbeddingResponseData(
                index=idx, embedding=embedding)
            data.append(embedding_data)

        request_id = f"embd-{random_uuid()}"
        created_time = int(time.monotonic())

        return protocol.EmbeddingResponse(id=request_id,
                                          created=created_time,
                                          model=req.model,
                                          data=data,
                                          usage=protocol.UsageInfo())
    except RuntimeError:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.name}' not found"})


@app.post("/v1/chat/completions")
async def chat(req: protocol.ChatCompletionRequest):
    options_key = {
        "temperature", "top_k", "top_p", "max_tokens", "presence_penalty",
        "frequency_penalty"
    }
    options = {
        k: v
        for k, v in req.model_dump().items()
        if k in options_key and v is not None
    }

    tool_dicts = None if req.tools is None else [
        tool.model_dump() for tool in req.tools
    ]

    try:
        response = await chat_client.chat(name=req.model,
                                          tools=tool_dicts,
                                          messages=req.messages,
                                          stream=req.stream,
                                          options=options)
    except Exception as e:
        traceback.print_exc()
        raise e

    if not inspect.isasyncgen(response):
        data = protocol.ChatCompletionResponse(
            **{
                "model":
                response.model,
                "choices": [
                    protocol.ChatCompletionResponseChoice(
                        **{
                            "index":
                            0,
                            "message":
                            protocol.ChatMessage(role="assistant",
                                                 content=response.content),
                            "finish_reason":
                            response.finish_reason
                        })
                ],
                "usage":
                protocol.UsageInfo(
                    prompt_tokens=response.prompt_tokens,
                    total_tokens=response.total_tokens,
                    completion_tokens=response.completion_tokens)
            })
        return data
    else:

        async def generate():
            async for rep in response:
                if isinstance(rep, ChatCompletionStreamResponseDone):
                    data = protocol.ChatCompletionStreamResponse(
                        **{
                            "model":
                            rep.model,
                            "choices": [
                                protocol.ChatCompletionResponseStreamChoice(
                                    **{
                                        "index": 0,
                                        "delta": protocol.DeltaMessage(),
                                        "finish_reason": rep.finish_reason
                                    })
                            ]
                        })
                    data = data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"
                    break
                else:
                    data = protocol.ChatCompletionStreamResponse(
                        **{
                            "model":
                            rep.model,
                            "choices": [
                                protocol.ChatCompletionResponseStreamChoice(
                                    **{
                                        "index":
                                        0,
                                        "delta":
                                        protocol.DeltaMessage(
                                            role="assistant",
                                            content=rep.delta_content)
                                    })
                            ]
                        })
                    data = data.model_dump_json(exclude_unset=True)
                    yield f"data: {data}\n\n"

        return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
