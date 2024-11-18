import json

import requests

messages = [{
    'role': 'user',
    'content': 'Why is the sky blue?',
}]

response = requests.post('http://localhost:8080/v1/chat/completions',
                         json={
                             "model": "Qwen/Qwen2.5-7B-Instruct",
                             "messages": messages,
                             "stream": True
                         })

for part in response.iter_lines():
    if not part.startswith(b"data:"):
        continue
    data = json.loads(part[6:])
    if 'content' in data["choices"][0]["delta"]:
        print(data["choices"][0]["delta"]['content'], end='', flush=True)

# end with a newline
print()
