class ConversableAgent(object):

    def __init__(self, name, description=""):
        self.name = name
        self.description = description

    def generate_reply(self, messages, stream=False, options=None):
        ...

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


Agent = ConversableAgent
