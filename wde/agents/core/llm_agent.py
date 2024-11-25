import inspect
from typing import Dict, List, Literal, Optional, Union

from wde.agents.core.chat_client import get_client
from wde.agents.core.conversable_agent import ConversableAgent
from wde.tasks.chat.schema.api import ChatCompletionStreamResponseDone


class LLMAgent(ConversableAgent):
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI Assistant."

    def __init__(self,
                 name: str,
                 system_message: Optional[Union[str, List]] = None,
                 llm_config: Optional[Union[Dict, Literal[False]]] = None,
                 description: Optional[str] = None):
        super().__init__(
            name, description if description is not None else system_message)

        self.llm_config = llm_config
        self._chat_client = get_client(self.llm_config)
        self._model_name = self._chat_client.model_name

        self._system_message = [{
            "content": system_message or self.DEFAULT_SYSTEM_MESSAGE,
            "role": "system"
        }]

    def generate_reply(self, messages, stream=False, options=None):
        if isinstance(messages, list):
            response = self._chat_client.chat(self._system_message + messages,
                                              None, stream, options)
        elif isinstance(messages, str):
            messages = [{"content": messages, "role": "user"}]
            response = self._chat_client.chat(self._system_message + messages,
                                              None, stream, options)

        if not inspect.isgenerator(response):
            return response.content
        else:

            def generator():
                for rep in response:
                    if not isinstance(rep, ChatCompletionStreamResponseDone):
                        yield rep.delta_content

            return generator()


RolePlayingAgent = LLMAgent