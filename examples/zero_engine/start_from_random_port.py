import os
import time

import numpy as np
from gevent.pool import Pool
from tqdm import tqdm

#########################################################
# Use random port to avoid conflicts

os.environ["WDE_NAME_SERVER_PORT"] = "19527"

from wde.engine.zero_engine import start_zero_engine
from wde.tasks.retriever.engine.client import RetrieverClient

model_name = "google-bert/bert-base-uncased"

#########################################################
# Start engine

engine_args = {"model": model_name}

server = start_zero_engine(engine_args)

###############################################################
# Wait until ready to use
client = RetrieverClient()

print("=" * 80)
print(f"Wait {model_name} available")
client.wait_service_available(model_name)

###############################################################
# Query basic information

print(client.get_services(model_name))

print("=" * 80)
print('support_methods')
print(client.support_methods(model_name))
print(client.info(model_name))

###############################################################
# encode

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

###############################################################
# Terminate engine

server.terminate()
