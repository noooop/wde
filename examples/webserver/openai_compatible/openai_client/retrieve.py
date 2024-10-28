from pprint import pprint

from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1/', api_key="empty")

pprint(client.models.retrieve('BAAI/bge-m3'))
pprint(client.models.retrieve('llama2:latest'))