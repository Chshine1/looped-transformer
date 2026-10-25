"""Rotary tensor operations."""

import torch
from torch import Tensor


def apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate split real/imaginary pairs without complex or half arithmetic."""
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        real, imag = x.to(dtype).chunk(2, dim=-1)
        cosine, sine = cosine.to(dtype), sine.to(dtype)
        return torch.cat(
            [real * cosine - imag * sine, real * sine + imag * cosine], dim=-1
        ).to(x.dtype)
