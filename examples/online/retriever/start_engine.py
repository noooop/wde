import time

import numpy as np
from gevent.pool import Pool
from tqdm import tqdm

from wde import const, envs
from wde.microservices.framework.zero_manager.client import ZeroManagerClient
from wde.microservices.standalone.deploy import ensure_zero_manager_available
from wde.tasks.retriever.engine.client import RetrieverClient

ensure_zero_manager_available()

#########################################################

model_name = "google-bert/bert-base-uncased"

engine_args = {"model": model_name}

#########################################################
# Start engine

manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
manager_client.wait_service_available(envs.ROOT_MANAGER_NAME)

model_name = engine_args["model"]

out = manager_client.start(name=model_name,
                           engine_kwargs={
                               "server_class": const.INFERENCE_ENGINE_CLASS,
                               "engine_args": engine_args
                           })
print("Start engine:", out)

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

out = manager_client.terminate(name=model_name)
print("Terminate engine:", out)
