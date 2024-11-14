import os
import time

from gevent.pool import Pool
from tqdm import tqdm

#########################################################
# Use random port to avoid conflicts

os.environ["WDE_NAME_SERVER_PORT"] = "19527"

from wde.client import RerankerClient
from wde.engine.zero_engine import start_zero_engine

model_name = "BAAI/bge-reranker-v2-m3"

#########################################################
# Start engine

engine_args = {"model": model_name}

server = start_zero_engine(engine_args)

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
# compute_score

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

server.terminate()
