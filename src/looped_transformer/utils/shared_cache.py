import torch
from torch import Tensor, nn


def apply_rotary(x: Tensor, cosine: Tensor, sine: Tensor) -> Tensor:
    """Rotate split real/imaginary pairs without complex or half arithmetic.

    Tables broadcast over x's paired last dimension. Keep FP64 inputs in
    FP64, promote low-precision activations to FP32, then restore their dtype.
    """
    dtype = torch.float64 if x.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=x.device.type, enabled=False):
        real, imag = x.to(dtype).chunk(2, dim=-1)
        cosine, sine = cosine.to(dtype), sine.to(dtype)
        return torch.cat(
            [real * cosine - imag * sine, real * sine + imag * cosine], dim=-1
        ).to(x.dtype)


class PositionalRoPECache(nn.Module):
    cosine: Tensor
    sine: Tensor

    def __init__(self, head_dimension: int, max_sequence_length: int):
        super().__init__()
        frequencies = 1.0 / (
            100.0 ** (torch.arange(0, head_dimension, 2).float() / head_dimension)
        )
        s = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(s, frequencies)
        self.register_buffer("cosine", frequencies.cos(), persistent=False)
        self.register_buffer("sine", frequencies.sin(), persistent=False)


class TimedMultiHeadRoPECache(nn.Module):
    cosine: Tensor
    sine: Tensor

    def __init__(self, heads_count: int, max_iterations: int, base_frequency: float):
        super().__init__()
        frequencies = 1.0 / (
            base_frequency ** (torch.arange(0, heads_count).float() / heads_count)
        )
        t = torch.arange(1, max_iterations + 1, dtype=torch.float32)
        frequencies = torch.outer(t, frequencies)
        self.register_buffer("cosine", frequencies.cos(), persistent=False)
        self.register_buffer("sine", frequencies.sin(), persistent=False)
