from wde.agents.core.conversable_agent import Agent


class UserInput(Agent):

    def __init__(self,
                 name="user",
                 prompt="user input: ",
                 description="user input"):
        super().__init__(name, description)
        self.prompt = prompt

    def generate_reply(self, messages, stream=False, options=None):
        content = input(self.prompt)
        return content
