import time

import shortuuid
import torch
from gevent.pool import Pool
from tqdm import tqdm

from wde.engine.gevent_engine import GeventLLMEngine

engine = GeventLLMEngine(model="google-bert/bert-base-uncased")

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 100


def worker(prompt):
    request_id = f"{shortuuid.random(length=22)}"
    outputs = engine.encode(inputs=prompt, request_id=request_id)
    return list(outputs)[0]


p = Pool(2)
out = []
for x in p.imap(worker, tqdm(prompts)):
    out.append(x.outputs)

print(torch.stack(out).shape)

time.sleep(1)

p = Pool(4)
out = []
for x in p.imap(worker, tqdm(prompts)):
    out.append(x.outputs)

print(torch.stack(out).shape)
