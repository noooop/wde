import torch
from vllm._custom_ops import copy_blocks
from vllm._custom_ops import \
    reshape_and_cache_flash as vllm_reshape_and_cache_flash
from vllm._custom_ops import swap_blocks

USE_VLLM_FLASH_ATTN = False

try:
    from vllm.vllm_flash_attn import _vllm_fa2_C  # noqa: F401
    USE_VLLM_FLASH_ATTN = True
except ImportError:
    pass

try:
    from vllm.vllm_flash_attn import _vllm_fa3_C  # noqa: F401
    USE_VLLM_FLASH_ATTN = True
except ImportError:
    pass

if USE_VLLM_FLASH_ATTN:

    def reshape_and_cache_flash(key: torch.Tensor, value: torch.Tensor,
                                key_cache: torch.Tensor,
                                value_cache: torch.Tensor,
                                slot_mapping: torch.Tensor,
                                kv_cache_dtype: str, k_scale: float,
                                v_scale: float):
        device = key.device

        k_scale_tensor = torch.tensor(k_scale, device=device)
        v_scale_tensor = torch.tensor(v_scale, device=device)
        return vllm_reshape_and_cache_flash(key, value, key_cache, value_cache,
                                            slot_mapping, kv_cache_dtype,
                                            k_scale_tensor, v_scale_tensor)
else:
    reshape_and_cache_flash = vllm_reshape_and_cache_flash

__all__ = ["swap_blocks", "copy_blocks", "reshape_and_cache_flash"]
