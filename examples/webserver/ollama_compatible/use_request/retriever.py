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
