import os
from typing import List, Optional

import shortuuid
import torch
from gevent.pool import Pool

from tests.tasks.utils import cleanup
from wde.engine.gevent_engine import GeventLLMEngine


class WDEGeventRunner:

    def __init__(self,
                 model_name: str,
                 max_num_requests: int = 4,
                 tokenizer_name: Optional[str] = None,
                 dtype: str = "half",
                 scheduling: str = "async",
                 attention_impl: Optional[str] = None,
                 **kwargs) -> None:
        if attention_impl is not None:
            os.environ["WDE_ATTENTION_BACKEND"] = attention_impl

        self.model = GeventLLMEngine(model=model_name,
                                     tokenizer=tokenizer_name,
                                     trust_remote_code=True,
                                     max_num_requests=max_num_requests,
                                     dtype=dtype,
                                     scheduling=scheduling,
                                     **kwargs)

    def encode(self, prompts: List[str]) -> List[List[float]]:

        def worker(prompt):
            request_id = f"{shortuuid.random(length=22)}"
            outputs = self.model.encode(inputs=prompt, request_id=request_id)
            return list(outputs)[0]

        outputs = []
        p = Pool(2)
        for out in p.imap(worker, prompts):
            embedding = out.outputs
            outputs.append(embedding)
        return outputs

    def compute_score(self, pairs) -> List[float]:

        def worker(pairs):
            request_id = f"{shortuuid.random(length=22)}"
            outputs = self.model.compute_score(inputs=pairs,
                                               request_id=request_id)
            return list(outputs)[0]

        outputs = []
        p = Pool(2)
        for out in p.imap(worker, pairs):
            score = out.score
            outputs.append(score)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        del self.model
        cleanup()


class WDEZERORunner:

    def __init__(self,
                 model_name: str,
                 max_num_requests: int = 4,
                 tokenizer_name: Optional[str] = None,
                 dtype: str = "half",
                 scheduling: str = "async",
                 attention_impl: Optional[str] = None,
                 **kwargs) -> None:
        if attention_impl is not None:
            os.environ["WDE_ATTENTION_BACKEND"] = attention_impl
        torch.multiprocessing.set_start_method('spawn', force=True)
        from wde.engine.zero_engine import start_zero_engine

        self.model_name = model_name

        engine_args = {
            "model": model_name,
            "tokenizer": tokenizer_name,
            "dtype": dtype,
            "max_num_requests": max_num_requests,
            "scheduling": scheduling,
        }
        engine_args.update(**kwargs)

        self.server = start_zero_engine(engine_args)

    def encode(self, prompts: List[str]) -> List[List[float]]:
        from wde.tasks.retriever.engine.client import RetrieverClient

        client = RetrieverClient()
        client.wait_service_available(self.model_name)

        def worker(prompt):
            output = client.encode(name=self.model_name, inputs=prompt)
            return output

        outputs = []
        p = Pool(2)
        for out in p.imap(worker, prompts):
            embedding = out.embedding
            outputs.append(embedding)
        return outputs

    def compute_score(self, pairs) -> List[float]:
        from wde.tasks.reranker.engine.client import RerankerClient

        client = RerankerClient()
        client.wait_service_available(self.model_name)

        def worker(pairs):
            output = client.compute_score(name=self.model_name, pairs=pairs)
            return output

        outputs = []
        p = Pool(2)
        for out in p.imap(worker, pairs):
            score = out.score
            outputs.append(score)
        return outputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.server.terminate()
