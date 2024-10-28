from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1/', api_key="empty")

print(
    client.embeddings.create(model="BAAI/bge-m3",
                             input="hello...",
                             encoding_format="base64"))
