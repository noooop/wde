
# [安装](https://github.com/noooop/wde/tree/main/setup)

# 使用 webserver

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