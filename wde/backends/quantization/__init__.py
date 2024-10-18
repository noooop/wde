from typing import Dict, Type

from wde.backends.quantization.aqlm import AQLMConfig
from wde.backends.quantization.awq import AWQConfig
from wde.backends.quantization.awq_marlin import AWQMarlinConfig
from wde.backends.quantization.base_config import QuantizationConfig
from wde.backends.quantization.bitsandbytes import BitsAndBytesConfig
from wde.backends.quantization.deepspeedfp import DeepSpeedFPConfig
from wde.backends.quantization.fbgemm_fp8 import FBGEMMFp8Config
from wde.backends.quantization.fp8 import Fp8Config
from wde.backends.quantization.gptq import GPTQConfig
from wde.backends.quantization.gptq_marlin import GPTQMarlinConfig
from wde.backends.quantization.gptq_marlin_24 import GPTQMarlin24Config
from wde.backends.quantization.marlin import MarlinConfig
from wde.backends.quantization.qqq import QQQConfig
from wde.backends.quantization.squeezellm import SqueezeLLMConfig

QUANTIZATION_METHODS: Dict[str, Type[QuantizationConfig]] = {
    "aqlm": AQLMConfig,
    "awq": AWQConfig,
    "deepspeedfp": DeepSpeedFPConfig,
    "fp8": Fp8Config,
    "fbgemm_fp8": FBGEMMFp8Config,
    # The order of gptq methods is important for config.py iteration over
    # override_quantization_method(..)
    "marlin": MarlinConfig,
    "gptq_marlin_24": GPTQMarlin24Config,
    "gptq_marlin": GPTQMarlinConfig,
    "awq_marlin": AWQMarlinConfig,
    "gptq": GPTQConfig,
    "squeezellm": SqueezeLLMConfig,
    "bitsandbytes": BitsAndBytesConfig,
    "qqq": QQQConfig,
}


def get_quantization_config(quantization: str) -> Type[QuantizationConfig]:
    if quantization not in QUANTIZATION_METHODS:
        raise ValueError(f"Invalid quantization method: {quantization}")
    return QUANTIZATION_METHODS[quantization]


__all__ = [
    "QuantizationConfig",
    "get_quantization_config",
    "QUANTIZATION_METHODS",
]
