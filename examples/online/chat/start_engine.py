import inspect

from wde import const, envs
from wde.client import ChatClient, ZeroManagerClient
from wde.microservices.standalone.deploy import ensure_zero_manager_available
from wde.tasks.chat.schema.api import ChatCompletionStreamResponseDone

ensure_zero_manager_available()

#########################################################

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

engine_args = {"model": model_name}

#########################################################
# Start engine

manager_client = ZeroManagerClient(envs.ROOT_MANAGER_NAME)
manager_client.wait_service_available(envs.ROOT_MANAGER_NAME)

model_name = engine_args["model"]

out = manager_client.start(name=model_name,
                           engine_kwargs={
                               "server_class": const.INFERENCE_ENGINE_CLASS,
                               "engine_args": engine_args
                           })
print("Start engine:", out)

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

out = manager_client.terminate(name=model_name)
print("Terminate engine:", out)