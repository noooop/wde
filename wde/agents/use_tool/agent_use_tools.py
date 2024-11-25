import shortuuid
from transformers.utils import get_json_schema

from wde.agents.core.llm_agent import LLMAgent
from wde.agents.use_tool.utils import validate_and_extract_tool_calls


class AgentUseTools(LLMAgent):
    DEFAULT_SYSTEM_MESSAGE = "You are a helpful AI assistant."

    def __init__(self,
                 name: str = "AgentUseTools",
                 system_message=DEFAULT_SYSTEM_MESSAGE,
                 tools=None,
                 llm_config=None,
                 description=None):
        super().__init__(name, system_message, llm_config, description)

        self.tools_schema = [get_json_schema(f) for f in tools]
        for t in self.tools_schema:
            t['function'].pop('return', None)

        self.tools = {
            s['function']['name']: f
            for s, f in zip(self.tools_schema, tools)
        }

    def generate_reply(self, messages, stream=False, options=None):
        if isinstance(messages, list):
            response = self._chat_client.chat(self._system_message + messages,
                                              self.tools_schema, stream,
                                              options)
        elif isinstance(messages, str):
            messages = [{"content": messages, "role": "user"}]
            response = self._chat_client.chat(self._system_message + messages,
                                              self.tools_schema, stream,
                                              options)

        validation_result, tool_calls, error_message = validate_and_extract_tool_calls(
            response.content)

        if not validation_result:
            raise RuntimeError("tool call error \n" + error_message)

        for i in range(len(tool_calls)):
            tool_calls[i]["id"] = shortuuid.random(6)

        messages.append({"role": "assistant", "tool_calls": tool_calls})

        for i in range(len(tool_calls)):
            try:
                tool_call = tool_calls[i]
                tool_call_id = tool_call["id"]
                name = tool_call["name"]
                arguments = tool_call["arguments"]
                content = self.tools[name](**arguments)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": name,
                    "content": str(content)
                })
            except Exception as e:
                print(e)

        response = self._chat_client.chat(self._system_message + messages,
                                          self.tools_schema, stream, options)
        return response.content
