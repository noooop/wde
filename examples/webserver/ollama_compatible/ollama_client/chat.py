from ollama import chat

messages = [
    {
        'role': 'user',
        'content': 'Why is the sky blue?',
    },
]

response = chat('Qwen/Qwen2.5-7B-Instruct', messages=messages)
print(response['message']['content'])
