import torch
from torch import Tensor


def apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    real, imag = x.chunk(2, dim=-1)
    return torch.cat(
        [real * cosine - imag * sine, real * sine + imag * cosine], dim=-1
    ).to(x.dtype)
