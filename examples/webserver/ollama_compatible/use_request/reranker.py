import requests

response = requests.post('http://localhost:11434/api/reranker',
                         json={
                             "model": "BAAI/bge-reranker-v2-m3",
                             "query": "query",
                             "passage": "passage",
                         })

print(response.json())
