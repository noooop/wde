import time

import numpy as np
from gevent.pool import Pool
from tqdm import tqdm

from wde.engine.zero_engine import start_zero_engine
from wde.tasks.retriever.engine.client import RetrieverClient

model_name = "google-bert/bert-base-uncased"

engine_args = {"model": model_name}

handle = start_zero_engine(engine_args)

client = RetrieverClient()

print("=" * 80)
print(f"Wait {model_name} available")
client.wait_service_available(model_name)
print(client.get_services(model_name))

print("=" * 80)
print('support_methods')
print(client.support_methods(model_name))
print(client.info(model_name))

prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
] * 100


def worker(prompt):
    client = RetrieverClient()

    output = client.encode(name=model_name, inputs=prompt)
    return output.embedding


p = Pool(2)
out = []
for embedding in p.imap(worker, tqdm(prompts)):
    out.append(embedding)

print(np.stack(out).shape)

time.sleep(1)

p = Pool(4)
out = []
for embedding in p.imap(worker, tqdm(prompts)):
    out.append(embedding)

print(np.stack(out).shape)

for h in handle:
    h.terminate()
