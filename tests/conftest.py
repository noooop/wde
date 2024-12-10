import random

import pytest

from tests.tasks.utils import HfRunner, WDERunner


@pytest.fixture(scope="session")
def wde_runner():
    return WDERunner


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner


@pytest.fixture(scope="session")
def example_prompts():
    prompts = [
        "JDK is developed by" * 16,
        "Birds can" * 16,
        "Hello, my name is" * 16,
        "The president of the United States is" * 16,
        "The capital of France is" * 16,
        "The future of AI is" * 16,
    ] * 11
    random.shuffle(prompts)
    return prompts
