import base64
import time
from typing import List, Literal, Union

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse

import wde.microservices.entrypoints.openai_compatible.schema as protocol
from wde.client import AsyncRerankerClient, AsyncRetrieverClient
from wde.utils import random_uuid

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


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
