from typing import Dict


class Workflow:
    EngineArgs: str
    Scheduler: str
    AttnBackend: str
    Tokenizer: str
    InputProcessor: str
    RequestProcessor: str
    OutputProcessor: str
    ModelInputBuilder: str
    Executor: str
    Worker: str

    @classmethod
    def from_engine(cls, engine):
        return cls()

    @classmethod
    def from_engine_args(cls, engine_args: Dict):
        return cls
