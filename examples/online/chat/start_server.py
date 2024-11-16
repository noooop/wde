import os

#########################################################
# Use random port to avoid conflicts

os.environ["WDE_NAME_SERVER_PORT"] = "19527"

import inspect

from wde.client import ChatClient
from wde.engine.zero_engine import start_zero_engine
from wde.tasks.chat.schema.api import ChatCompletionStreamResponseDone

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

#########################################################
# Start engine

engine_args = {"model": model_name}

server = start_zero_engine(engine_args)

###############################################################
# Wait until ready to use
client = ChatClient()

print("=" * 80)
print(f"Wait {model_name} available")
client.wait_service_available(model_name)

###############################################################
# Query basic information

print(client.get_services(model_name))

print("=" * 80)
print('support_methods')
print(client.support_methods(model_name))
print(client.info(model_name))

###############################################################
# generate


def chat_demo(stream):

    prompt = "给我介绍一下大型语言模型。"

    messages = [{"role": "user", "content": prompt}]

    response = client.chat(model_name,
                           messages,
                           options={"max_tokens": 512},
                           stream=stream)

    if inspect.isgenerator(response):
        for part in response:
            if isinstance(part, ChatCompletionStreamResponseDone):
                print()
                print("completion_tokens:", part.completion_tokens)
            else:
                print(part.delta_content, end="", flush=True)
    else:
        print(response.content)
        print("completion_tokens:", response.completion_tokens)


chat_demo(stream=False)
chat_demo(stream=True)

###############################################################
# Terminate engine

server.terminate()
