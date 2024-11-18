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
                                            }])

print(completion.choices[0].message.content)
