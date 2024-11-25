from wde.agents.core.llm_agent import LLMAgent


class SummaryAgent(LLMAgent):
    DEFAULT_SYSTEM_MESSAGE = "Summarize the takeaway from the conversation. Do not add any introductory phrases."

    def __init__(self,
                 name: str = "SummaryAgent",
                 system_message=DEFAULT_SYSTEM_MESSAGE,
                 llm_config=None,
                 description=None):
        super().__init__(name, system_message, llm_config, description)

    def summary(self, session, summary_prompt=None):
        return session.summary(self, summary_prompt)
