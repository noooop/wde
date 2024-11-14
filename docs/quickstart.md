
# 安装

```commandline
pip install -r requirements.txt
pip install https://github.com/noooop/wde/archive/refs/heads/main.zip
```

建议使用conda配置文件一键安装 [请移步](https://github.com/noooop/wde/tree/main/setup)

# 离线批量推理 (Offline Batched Inference)

## Retriever(Embeddings) 模型

```python
from wde import LLM

prompts = [
    "prompt1", "prompt2",
]

llm = LLM(model='BAAI/bge-m3')

outputs = llm.encode(prompts)

for output in outputs:
    print(output.outputs.shape)
```

## Reranker 模型

```python
from wde import LLM

pairs = [['query', 'passage']]

llm = LLM(model="BAAI/bge-reranker-v2-m3")

outputs = llm.compute_score(pairs)
for output in outputs:
    print(output.score)
```

[支持的模型](https://github.com/noooop/wde/blob/main/docs/supported_models.md)
 
更多示例 [请移步](https://github.com/noooop/wde/blob/main/examples/offline/)


# 在线推理 (Serving

wde 自带一个微服务框架，可以在一台机器上部署多个模型，随时启动和停止单独模型

## server 常用命令

```commandline
wde --help
```

```
Usage: wde [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  deploy
  server
  serving
  start
  terminate
```

1. 首先启动 server

```commandline
wde server
```

> 服务发现默认使用 9527 端口，可以使用 WDE_NAME_SERVER_PORT 设定一个没有占用的端口避免冲突，客户端同样需要设置为相同端口才能连上

2. 启动模型

server 窗口已经占用了，需要另外一个窗口运行下面命令

```commandline
wde start BAAI/bge-m3
wde start BAAI/bge-reranker-v2-m3
```

下载模型比较慢，建议先下载好模型再运行模型

```commandline
huggingface-cli download BAAI/bge-m3
huggingface-cli download BAAI/bge-reranker-v2-m3
```

3. 停止模型 
```commandline
wde terminate BAAI/bge-m3
wde terminate BAAI/bge-reranker-v2-m3
```

4. 使用部署文件一次性部署多个模型
```commandline
wde deploy examples/online/deploy.yml
```

> wde deploy 是将模型部署命令提交到 server， 所以要保持另外一个窗口的 wde server 一直运行

[示例部署文件](https://github.com/noooop/wde/blob/main/examples/online/deploy.yml)

5. 可以使用 api 在代码里启动和停止模型，等待模型加载完成

```python
from wde import const, envs
from wde.client import ZeroManagerClient

manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
manager_client.wait_service_available(envs.ROOT_MANAGER_NAME)

model_name = "google-bert/bert-base-uncased"

engine_args = {"model": model_name}

#########################################################
# Start engine
out = manager_client.start(name=model_name,
                           engine_kwargs={
                               "server_class": const.INFERENCE_ENGINE_CLASS,
                               "engine_args": engine_args
                           })
print("Start engine:", out)

###############################################################
# Terminate engine

out = manager_client.terminate(name=model_name)
print("Terminate engine:", out)

```

详细代码 [retriever](https://github.com/noooop/wde/blob/main/examples/online/retriever/start_engine.py) [reranker](https://github.com/noooop/wde/blob/main/examples/online/reranker/start_engine.py)

6. 一键启动模型

```commandline
wde serving examples/online/deploy.yml
```

serving 命令相当于 server + deploy 命令，启动服务并部署模型


## 使用 zeromq 客户端

zeromq 相比 http 吞吐高延迟小，尤其是传输 Embeddings，建议优先使用 zeromq 客户端

可以使用 gevent 和 asyncio 并发请求，下面以 gevent 为例，展示大概代码框架

详细代码 [retriever](https://github.com/noooop/wde/blob/main/examples/online/retriever/start_engine.py) [reranker](https://github.com/noooop/wde/blob/main/examples/online/reranker/start_engine.py)


```python
import numpy as np
from gevent.pool import Pool
from wde.client import RetrieverClient

client = RetrieverClient()

model_name = "BAAI/bge-m3"

def worker(prompt):
    output = client.encode(name=model_name, inputs=prompt)
    return output.embedding


prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]

p = Pool(2)
out = []
for embedding in p.imap(worker, prompts):
    out.append(embedding)

print(np.stack(out).shape)

```

## 使用 webserver

项目自带与 ollama 和 openai 兼容的 webserver, 可以使用下面的部署文件一键部署

```commandline
wde serving examples/webserver/deploy.yml
```

[webserver 示例部署文件](https://github.com/noooop/wde/blob/main/examples/webserver/deploy.yml)
[ollama 和 openai 客户端示例](https://github.com/noooop/wde/tree/main/examples/webserver)

使用 requests 调用 ollama 兼容客户端代码如下

```python
import requests

response = requests.get('http://localhost:11434/api/tags')
print(response.json())

response = requests.post('http://localhost:11434/api/embeddings',
                         json={
                             "model": "BAAI/bge-m3",
                             "prompt": "hello..."
                         })

print(response.json().keys())
print(len(response.json()["embedding"]))
```

## all-in-one

有时候需要一个程序同时启动服务端和客户端，下面示例可以满足你。

> 注意使用 WDE_NAME_SERVER_PORT 设定一个没有占用的端口避免冲突

详细代码 [retriever](https://github.com/noooop/wde/blob/main/examples/online/retriever/start_server.py) [reranker](https://github.com/noooop/wde/blob/main/examples/online/reranker/start_server.py)


