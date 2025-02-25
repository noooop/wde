from typing import Dict, Type

from wde.utils import LazyLoader, lazy_import
from wde.workflows.core.backends.quantization.base_config import \
    QuantizationConfig

_QUANTIZATION_METHODS: Dict[str, str] = {
    "aqlm": "wde.workflows.core.backends.quantization.aqlm:AQLMConfig",
    "awq": "wde.workflows.core.backends.quantization.awq:AWQConfig",
    "deepspeedfp":
    "wde.workflows.core.backends.quantization.deepspeedfp:DeepSpeedFPConfig",
    "fp8": "wde.workflows.core.backends.quantization.fp8:Fp8Config",
    "fbgemm_fp8":
    "wde.workflows.core.backends.quantization.fbgemm_fp8:FBGEMMFp8Config",
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin": "wde.workflows.core.backends.quantization.marlin:MarlinConfig",
    "gptq_marlin_24":
    "wde.workflows.core.backends.quantization.gptq_marlin_24:GPTQMarlin24Config",
    "gptq_marlin":
    "wde.workflows.core.backends.quantization.gptq_marlin:GPTQMarlinConfig",
    "awq_marlin":
    "wde.workflows.core.backends.quantization.awq_marlin:AWQMarlinConfig",
    "gptq": "wde.workflows.core.backends.quantization.gptq:GPTQConfig",
    "squeezellm":
    "wde.workflows.core.backends.quantization.squeezellm:SqueezeLLMConfig",
    "bitsandbytes":
    "wde.workflows.core.backends.quantization.bitsandbytes:BitsAndBytesConfig",
    "qqq": "wde.workflows.core.backends.quantization.qqq:QQQConfig",
}

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    _name: LazyLoader(_module)
    for _name, _module in _QUANTIZATION_METHODS.items()
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in _QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return lazy_import(_QUANTIZATION_METHODS[quantization])


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
