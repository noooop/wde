import inspect

from wde.client import ChatClient
from wde.tasks.chat.schema.api import (ChatCompletionResponse,
                                       ChatCompletionStreamResponse,
                                       ChatCompletionStreamResponseDone)


class ZeroChatClient:

    def __init__(self, model, options=None):
        self.options = options or {}
        self._chat_client = ChatClient()
        self.model_name = model

    def chat(self, messages, tools=None, stream=False, options=None):
        _options = dict(**self.options)
        _options.update(options or {})
        response = self._chat_client.chat(self.model_name, messages, tools,
                                          stream, _options)
        return response


class OpenAiChatClient:

    def __init__(self,
                 model,
                 options=None,
                 base_url="http://localhost:8080",
                 api_key="empty",
                 **kwargs):
        from gevent import monkey
        monkey.patch_socket()

        from openai import OpenAI, Stream
        self.options = options or {}

        self.Stream = Stream
        self._chat_client = OpenAI(base_url=base_url,
                                   api_key=api_key,
                                   **kwargs)
        self.model_name = model

    def chat(self, messages, tools=None, stream=False, options=None):
        _options = dict(**self.options)
        _options.update(options or {})

        response = self._chat_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            stream=stream,
            tools=tools,
            **_options)

        if not isinstance(response, self.Stream):
            return ChatCompletionResponse(
                **{
                    "model": self.model_name,
                    "content": response.choices[0].message.content,
                    "finish_reason": response.choices[0].finish_reason,
                    "completion_tokens": response.usage.completion_tokens,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "total_tokens": response.usage.total_tokens
                })
        else:

            def generator():

                for res in response:
                    res = res.choices[0]
                    if res.finish_reason is None:
                        yield ChatCompletionStreamResponse(
                            **{
                                "model": self.model_name,
                                "delta_content": res.delta.content
                            })
                    else:
                        yield ChatCompletionStreamResponseDone(
                            **{
                                "model": self.model_name,
                                "finish_reason": res.finish_reason,
                                "completion_tokens": 0,
                                "prompt_tokens": 0,
                                "total_tokens": 0
                            })

            return generator()


class OllamaChatClient:

    def __init__(self, model, options=None, base_url=None, **kwargs):
        from gevent import monkey
        monkey.patch_socket()

        from ollama import Client
        self.options = options or {}
        self._chat_client = Client(host=base_url, **kwargs)
        self.model_name = model

    def chat(self, messages, tools=None, stream=False, options=None):
        _options = dict(**self.options)
        _options.update(options or {})

        response = self._chat_client.chat(model=self.model_name,
                                          messages=messages,
                                          stream=stream,
                                          options=_options)

        if not inspect.isgenerator(response):
            return ChatCompletionResponse(
                **{
                    "model":
                    self.model_name,
                    "content":
                    response['message']['content'],
                    "finish_reason":
                    response['done_reason'],
                    "completion_tokens":
                    response.get('eval_count', 0),
                    "prompt_tokens":
                    response.get('prompt_eval_count', 0),
                    "total_tokens":
                    response.get('eval_count', 0) +
                    response.get('prompt_eval_count', 0)
                })
        else:

            def generator():
                for res in response:
                    if not res["done"]:
                        yield ChatCompletionStreamResponse(
                            **{
                                "model": self.model_name,
                                "delta_content": res['message']['content']
                            })
                    else:
                        yield ChatCompletionStreamResponseDone(
                            **{
                                "model":
                                self.model_name,
                                "finish_reason":
                                res['done_reason'],
                                "completion_tokens":
                                res.get('eval_count', 0),
                                "prompt_tokens":
                                res.get('prompt_eval_count', 0),
                                "total_tokens":
                                res.get('eval_count', 0) +
                                res.get('prompt_eval_count', 0)
                            })

            return generator()


def get_client(llm_config=None):
    import copy

    llm_config = copy.deepcopy(llm_config)
    llm_client_type = llm_config.pop("type")

    if llm_client_type == "zeroclient":
        client = ZeroChatClient(**llm_config)
    elif llm_client_type == "openai":
        client = OpenAiChatClient(**llm_config)
    elif llm_client_type == "ollama":
        client = OllamaChatClient(**llm_config)
    else:
        raise KeyError(f"llm client type {llm_client_type} not support")

    return client
