from typing import Optional

import torch

from .interface import Platform, PlatformEnum, UnspecifiedPlatform

current_platform: Optional[Platform]

if torch.version.cuda is not None:
    from .cuda import CudaPlatform
    current_platform = CudaPlatform()
else:
    current_platform = UnspecifiedPlatform()

__all__ = ['Platform', 'PlatformEnum', 'current_platform']
