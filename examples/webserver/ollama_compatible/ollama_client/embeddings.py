from ollama import embeddings

response = embeddings("BAAI/bge-m3", prompt="hello...")
print(response["embedding"])
