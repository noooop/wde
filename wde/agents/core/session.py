from wde.agents.core.conversable_agent import Agent


class MSG(object):
    role: Agent
    content: str

    def __init__(self, role, content):
        self.role = role
        self.content = content

    @classmethod
    def from_any(cls, msg):
        if isinstance(msg, MSG):
            role, content = msg.role, msg.content

        elif isinstance(msg, (list, tuple)):
            role, content = msg
        elif isinstance(msg, dict):
            role = msg["role"]
            content = msg["content"]
        else:
            raise RuntimeError(f"MSG.from_any do not support [{msg}]")

        return MSG(role, content)


class ViolationOFChatOrder(Exception):
    pass


class Session(object):
    DEFAULT_SUMMARY_PROMPT = "Summarize the takeaway from the conversation. Do not add any introductory phrases."

    def __init__(self,
                 participants,
                 history=None,
                 termination_msg=None,
                 is_termination_msg=None):
        self.participants = participants
        self.history = list()

        self.termination_msg = termination_msg
        self.is_termination_msg = is_termination_msg

        if history is not None:
            self.extend(history)

    @property
    def n_participants(self):
        return len(self.participants)

    def chat_order(self):
        last_role = self.history[-1].role
        order = self.participants + self.participants
        index = order.index(last_role) + 1
        return order[index:index + len(self.participants)]

    def append(self, msg):
        msg = MSG.from_any(msg)

        if len(self.history) == 0:
            self.history.append(msg)
        else:
            chat_order = self.chat_order()
            next = chat_order[0]
            if msg.role != next:
                raise ViolationOFChatOrder(
                    f"{msg.role.name} wait for {next.name} speak. content: [{msg.content}]"
                )
            else:
                self.history.append(msg)

    def extend(self, history, start=0):
        if len(self.history) == 0:
            self.history.append(MSG.from_any(history[0]))
            self.extend(history, start=1)
        else:
            chat_order = self.chat_order()
            for i, msg in enumerate(history[start:]):
                msg = MSG.from_any(msg)

                next = chat_order[i % self.n_participants]
                if msg.role != next:
                    raise ViolationOFChatOrder(
                        f"{msg.role.name} @ {i+start} wait for {next.name} speak. content: [{msg.content}]"
                    )
                else:
                    self.history.append(msg)

    def messages_for_role(self, role):
        chat_order = self.chat_order()
        next = chat_order[0]

        if role != next:
            raise ViolationOFChatOrder(
                f"{role.name} wait for {next.name} speak")

        messages = []
        for msg in self.history:
            if msg.role == role:
                messages.append({"content": msg.content, "role": "assistant"})
            else:
                messages.append({"content": msg.content, "role": "user"})
        return messages

    def chat_messages_for_summary(self):
        prompt = ""
        for msg in self.history:
            prompt += f"{msg.role.name} : \n\n"
            prompt += f"{msg.content} : \n\n"
        return prompt

    def summary(self, agent, summary_prompt=DEFAULT_SUMMARY_PROMPT):
        history_prompt = self.chat_messages_for_summary()
        if summary_prompt:
            history_prompt = history_prompt + summary_prompt
        return agent.generate_reply(history_prompt)

    def chat(self, max_turns=None, verbose=True, verbose_history=True):
        if len(self.history) == 0:
            raise ViolationOFChatOrder("no init chat")

        def _verbose(msg, i):
            print(msg.role,
                  f"(round {0 if i == 0 else (i-1) // self.n_participants})")
            print()
            print(msg.content)
            print()
            if i != 0 and i % self.n_participants == 0:
                print("=" * 80)

        if verbose_history:
            for i, msg in enumerate(self.history):
                _verbose(msg, i)

        max_turns = 100000000 if max_turns is None else max_turns

        ii = self.n_participants - len(self.history) % self.n_participants
        chat_order = self.chat_order()
        for role in chat_order[:ii]:
            content = role.generate_reply(self.messages_for_role(role))
            self.history.append(MSG(role, content))

            if verbose:
                _verbose(self.history[-1], len(self.history))

            if isinstance(self.termination_msg, str):
                if self.termination_msg.lower() in content.lower():
                    return

            if callable(self.is_termination_msg):
                if self.is_termination_msg(content):
                    return

        chat_order = self.chat_order()
        for i in range(max_turns - 1):
            for role in chat_order:
                content = role.generate_reply(self.messages_for_role(role))
                self.history.append(MSG(role, content))

                if verbose:
                    _verbose(self.history[-1], len(self.history))

                if isinstance(self.termination_msg, str):
                    if self.termination_msg.lower() in content.lower():
                        return

                if callable(self.is_termination_msg):
                    if self.is_termination_msg(content):
                        return


if __name__ == '__main__':

    def test_ViolationOFChatOrder():
        A = Agent("A")
        B = Agent("B")
        C = Agent("C")

        s = Session(participants=[A, B])
        s.append((A, "a1"))
        s.append((B, "b1"))
        try:
            s.append((B, "b2"))
        except ViolationOFChatOrder as e:
            print(e)

        s = Session(participants=[A, B])
        try:
            s.extend([
                (A, "a1"),
                (B, "b1"),
                (A, "a2"),
                (B, "b2"),
                (A, "a3"),
                (B, "b3"),
                (A, "a4"),
                (A, "a5"),
            ])
        except ViolationOFChatOrder as e:
            print(e)

        s = Session(participants=[A, B, C])
        try:
            s.extend([
                (A, "a1"),
                (B, "b1"),
                (A, "a2"),
            ])
        except ViolationOFChatOrder as e:
            print(e)
