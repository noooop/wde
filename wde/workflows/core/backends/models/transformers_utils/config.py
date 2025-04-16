from pathlib import Path
from typing import Optional, Union

from transformers import PretrainedConfig
from vllm.transformers_utils.config import get_config as get_config_vllm
from vllm.transformers_utils.config import (get_hf_text_config,
                                            try_get_generation_config)
from vllm.transformers_utils.utils import maybe_model_redirect


def get_config(
    model: Union[str, Path],
    trust_remote_code: bool,
    revision: Optional[str] = None,
    code_revision: Optional[str] = None,
    **kwargs,
) -> PretrainedConfig:
    model = maybe_model_redirect(model)
    return get_config_vllm(model, trust_remote_code, revision, code_revision,
                           **kwargs)


__all__ = ["get_config", "try_get_generation_config", "get_hf_text_config"]
