import datetime

from fastapi import FastAPI
from fastapi.responses import JSONResponse

import wde.microservices.entrypoints.ollama_compatible.schema as protocol
from wde.client import AsyncRerankerClient, AsyncRetrieverClient

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
    response = await retriever_client.get_service_names()
    services = response.msg["service_names"]

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
        rep = await retriever_client.info(name)
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
        rep = await retriever_client.encode(req.model, req.prompt, req.options)
        return {"model": req.model, "embedding": rep.embedding.tolist()}
    except RuntimeError:
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{req.model}' not found"})


@app.post("/api/reranker")
async def reranker(req: protocol.RerankerRequest):
    try:
        rep = await reranker_client.compute_score(req.model,
                                                  (req.query, req.passage),
                                                  req.options)
        return {"model": req.model, "score": rep.score}
    except RuntimeError:
        return JSONResponse(
            status_code=404,
            content={"error": f"model '{req.model}' not found"})


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=11434)
