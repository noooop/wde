import requests

messages = [{
    'role': 'user',
    'content': 'Why is the sky blue?',
}]

response = requests.post('http://localhost:8080/v1/chat/completions',
                         json={
                             "model": "Qwen/Qwen2.5-7B-Instruct",
                             "messages": messages,
                             "stream": False
                         })

print(response.json())
