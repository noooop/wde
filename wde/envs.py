import os
from typing import Any, Callable, Dict

environment_variables: Dict[str, Callable[[], Any]] = {
    "LOGGING_LEVEL":
    lambda: os.getenv("WDE_LOGGING_LEVEL", "INFO"),
    "USE_MODELSCOPE":
    lambda: os.environ.get("WDE_USE_MODELSCOPE", "False").lower() == "true",
    "ATTENTION_BACKEND":
    lambda: os.getenv("WDE_ATTENTION_BACKEND", None),
    "ENGINE_ITERATION_TIMEOUT_S":
    lambda: int(os.environ.get("WDE_ENGINE_ITERATION_TIMEOUT_S", "60")),
    "ROOT_MANAGER_NAME":
    lambda: os.environ.get("WDE_ROOT_MANAGER_NAME", "RootZeroManager"),
    "NAME_SERVER_PORT":
    lambda: int(os.environ.get("WDE_NAME_SERVER_PORT", 9527)),
}


def __getattr__(name: str):
    # lazy evaluation of environment variables
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(environment_variables.keys())
