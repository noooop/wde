import requests

response = requests.post('http://localhost:8080/v1/embeddings',
                         json={
                             "model": "BAAI/bge-m3",
                             "input": "hello...",
                             "encoding_format": "float"
                         })

print(response.json().keys())
print(len(response.json()["data"][0]["embedding"]))
