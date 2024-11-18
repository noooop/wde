import json

import requests

messages = [{
    'role': 'user',
    'content': 'Why is the sky blue?',
}]

response = requests.post('http://localhost:11434/api/chat',
                         json={
                             "model": "Qwen/Qwen2.5-7B-Instruct",
                             "messages": messages,
                             "stream": True
                         })

for part in response.iter_lines():
    print(json.loads(part)['message']['content'], end='', flush=True)

# end with a newline
print()
