import time

from gevent.pool import Pool
from tqdm import tqdm

from wde import const, envs
from wde.microservices.framework.zero_manager.client import ZeroManagerClient
from wde.microservices.standalone.deploy import ensure_zero_manager_available
from wde.tasks.reranker.engine.client import RerankerClient

ensure_zero_manager_available()

#########################################################

model_name = "BAAI/bge-reranker-v2-m3"

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

client = RerankerClient()

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

pairs_list = [['query', 'passage'], ['what is panda?', 'hi'],
              [
                  'what is panda?',
                  'The giant panda (Ailuropoda melanoleuca), '
                  'sometimes called a panda bear or simply panda, '
                  'is a bear species endemic to China.'
              ]] * 100


def worker(pairs):
    output = client.compute_score(name=model_name, pairs=pairs)
    return output.score


p = Pool(2)
out = []
for score in p.imap(worker, tqdm(pairs_list)):
    out.append(score)

print(len(out))

time.sleep(1)

p = Pool(4)
out = []
for score in p.imap(worker, tqdm(pairs_list)):
    out.append(score)

print(len(out))

###############################################################
# Terminate engine

out = manager_client.terminate(name=model_name)
print("Terminate engine:", out)
