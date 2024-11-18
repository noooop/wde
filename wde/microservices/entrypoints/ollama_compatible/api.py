import datetime
import inspect
import json

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse

import wde.microservices.entrypoints.ollama_compatible.schema as protocol
from wde.client import (AsyncChatClient, AsyncRerankerClient,
                        AsyncRetrieverClient)
from wde.tasks.chat.schema.api import ChatCompletionStreamResponseDone

chat_client = AsyncChatClient()
retriever_client = AsyncRetrieverClient()
reranker_client = AsyncRerankerClient()
app = FastAPI()


def get_timestamp():
    return datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%d %H:%M:%S.%f')


@app.get("/")
async def health():
    return "wde is running"


@app.get("/api/tags")
async def tags():
    response = await chat_client.get_service_names()
    services = response.msg["service_names"]
    response = await retriever_client.get_service_names()
    services += response.msg["service_names"]

    models = []
    for s in services:
        models.append({
            'name': s,
            'model': s,
            'modified_at': "",
            'size': "",
            'digest': "",
            "details": {}
        })

    return {"models": models}


@app.post("/api/show")
async def show(req: protocol.ShowRequest):
    try:
        name = req.name
        rep = await chat_client.info(name)
        details = rep.msg
        out = {
            'name': name,
            'model': name,
            'modified_at': "",
            'size': "",
            'digest': "",
            "details": details
        }
        return out
    except Exception:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.name}' not found"})


@app.post("/api/embeddings")
async def embeddings(req: protocol.EmbeddingsRequest):
    try:
        response = await retriever_client.encode(req.model, req.prompt,
                                                 req.options)
        return {"model": req.model, "embedding": response.embedding.tolist()}
    except RuntimeError:
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{req.model}' not found"})


@app.post("/api/reranker")
async def reranker(req: protocol.RerankerRequest):
    try:
        response = await reranker_client.compute_score(
            req.model, (req.query, req.passage), req.options)
        return {"model": response.model, "score": response.score}
    except RuntimeError:
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{req.model}' not found"})


@app.post("/api/chat")
async def chat(req: protocol.ChatCompletionRequest):
    try:
        response = await chat_client.chat(name=req.model,
                                          messages=req.messages,
                                          stream=req.stream,
                                          options=req.options)
        if not inspect.isasyncgen(response):
            data = {
                "model": response.model,
                "created_at": get_timestamp(),
                "message": {
                    "role": "assistant",
                    "content": response.content
                },
                "done": True,
                "done_reason": response.finish_reason,
                "eval_count": response.completion_tokens,
                "prompt_eval_count": response.prompt_tokens
            }
            return data
        else:

            async def generate():
                async for rep in response:
                    if isinstance(rep, ChatCompletionStreamResponseDone):
                        data = json.dumps({
                            "model":
                            req.model,
                            "created_at":
                            get_timestamp(),
                            "message": {
                                "role": "assistant",
                                "content": ""
                            },
                            "done":
                            True,
                            "done_reason":
                            rep.finish_reason,
                            "eval_count":
                            rep.completion_tokens,
                            "prompt_eval_count":
                            rep.prompt_tokens
                        })
                        yield data
                        yield "\n"
                        break
                    else:
                        delta_content = rep.delta_content
                        data = json.dumps({
                            "model": req.model,
                            "created_at": get_timestamp(),
                            "message": {
                                "role": "assistant",
                                "content": delta_content
                            },
                            "done": False
                        })
                        yield data
                        yield "\n"

            return StreamingResponse(generate(),
                                     media_type="application/x-ndjson")
    except RuntimeError:
        return JSONResponse(status_code=404,
                            content={"error": f"model '{req.name}' not found"})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
