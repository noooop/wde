# ruff: noqa: F841

# Adapted from
# https://github.com/vllm-project/flash-attention/blob/main/vllm_flash_attn/flash_attn_interface.py

from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache)
from vllm.vllm_flash_attn.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version)

__all__ = [
    "flash_attn_varlen_func", "flash_attn_with_kvcache",
    "get_flash_attn_version", "flash_attn_supports_fp8"
]
