from ollama import show

response = show("BAAI/bge-m3")
print(response)
response = show('llama2:latest')
print(response)
