"""Workflow Defined Engine"""

from .utils import LazyLoader
from .version import __version__

_modules = {
    "LLM": "wde.engine.offline:LLM",
    "LLMEngine": "wde.workflows.core.llm_engine:LLMEngine",
    "ModelRegistry": "wde.workflows.core.modelzoo:ModelRegistry",
    "TextPrompt": "wde.workflows.core.schema.engine_io:TextPrompt",
    "SamplingParams": "wde.workflows.decoding:SamplingParams",
}

for _name, _module in _modules.items():
    globals()[_name] = LazyLoader(_module)

__version__ = __version__
