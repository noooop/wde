import pytest

from tests.tasks.utils import HfRunner, WDERunner


@pytest.fixture(scope="session")
def wde_runner():
    return WDERunner


@pytest.fixture(scope="session")
def hf_runner():
    return HfRunner
