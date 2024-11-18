from openai import OpenAI

client = OpenAI(base_url='http://localhost:8080/v1/', api_key="empty")

completion = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
                                            messages=[{
                                                "role":
                                                "system",
                                                "content":
                                                "You are a helpful assistant."
                                            }, {
                                                "role":
                                                "user",
                                                "content":
                                                "Why is the sky blue?"
                                            }],
                                            stream=True)

for chunk in completion:
    print(chunk.choices[0].delta.content, end="", flush=True)
print()
