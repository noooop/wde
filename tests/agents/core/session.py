import pytest

from wde.agents.core.conversable_agent import Agent
from wde.agents.core.session import Session, ViolationOFChatOrder


def test_ViolationOFChatOrder1():
    A = Agent("A")
    B = Agent("B")

    s = Session(participants=[A, B])
    s.append((A, "a1"))
    s.append((B, "b1"))

    with pytest.raises(ViolationOFChatOrder):
        s.append((B, "b2"))


def test_ViolationOFChatOrder2():
    A = Agent("A")
    B = Agent("B")
    s = Session(participants=[A, B])

    with pytest.raises(ViolationOFChatOrder):
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


def test_ViolationOFChatOrder3():
    A = Agent("A")
    B = Agent("B")
    C = Agent("C")

    s = Session(participants=[A, B, C])
    with pytest.raises(ViolationOFChatOrder):
        s.extend([
            (A, "a1"),
            (B, "b1"),
            (A, "a2"),
        ])
