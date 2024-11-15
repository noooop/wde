from dataclasses import dataclass, fields
from typing import List, Optional, Union

from wde.logger import init_logger
from wde.workflows.core.config import (DeviceConfig, EngineConfig, LoadConfig,
                                       ModelConfig)

logger = init_logger(__name__)


def nullable_str(val: str):
    if not val or val == "None":
        return None
    return val


@dataclass
class EngineArgs:
    model: str
    served_model_name: Optional[Union[List[str]]] = None
    tokenizer: Optional[str] = None
    tokenizer_mode: str = 'auto'
    trust_remote_code: bool = False

    dtype: str = 'auto'
    seed: int = 0

    revision: Optional[str] = None
    code_revision: Optional[str] = None
    rope_scaling: Optional[dict] = None
    rope_theta: Optional[float] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    quantization_param_path: Optional[str] = None

    download_dir: Optional[str] = None
    load_format: str = 'auto'
    model_loader_extra_config: Optional[dict] = None
    ignore_patterns: Optional[Union[str, List[str]]] = None

    disable_sliding_window: bool = False
    max_model_len: Optional[int] = None

    device: str = 'auto'

    def create_engine_config(self):
        device_config = DeviceConfig(device=self.device)
        model_config = ModelConfig(
            model=self.model,
            tokenizer=self.tokenizer,
            tokenizer_mode=self.tokenizer_mode,
            trust_remote_code=self.trust_remote_code,
            dtype=self.dtype,
            seed=self.seed,
            revision=self.revision,
            code_revision=self.code_revision,
            rope_scaling=self.rope_scaling,
            rope_theta=self.rope_theta,
            tokenizer_revision=self.tokenizer_revision,
            max_model_len=self.max_model_len,
            quantization=self.quantization,
            quantization_param_path=self.quantization_param_path,
            disable_sliding_window=self.disable_sliding_window,
            served_model_name=self.served_model_name)
        load_config = LoadConfig(
            load_format=self.load_format,
            download_dir=self.download_dir,
            model_loader_extra_config=self.model_loader_extra_config,
            ignore_patterns=self.ignore_patterns,
        )

        return EngineConfig(model_config=model_config,
                            device_config=device_config,
                            load_config=load_config)

    def to_dict(self):
        return dict(
            (field.name, getattr(self, field.name)) for field in fields(self))
