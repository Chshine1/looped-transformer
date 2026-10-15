import torch
from torch import nn, Tensor


class PositionalRoPECache(nn.Module):
    rotations: Tensor

    def __init__(self, head_dimension: int, max_sequence_length: int):
        super().__init__()
        frequencies = 1.0 / (
            100.0 ** (torch.arange(0, head_dimension, 2).float() / head_dimension)
        )
        s = torch.arange(max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(s, frequencies)
        rotations = torch.polar(torch.ones_like(frequencies), frequencies)

        self.register_buffer("rotations", rotations, persistent=False)


class TimedMultiHeadRoPECache(nn.Module):
    rotations: Tensor

    def __init__(self, heads_count: int, max_iterations: int, base_frequency: float):
        super().__init__()
        frequencies = 1.0 / (
            base_frequency ** (torch.arange(0, heads_count).float() / heads_count)
        )
        t = torch.arange(1, max_iterations + 1, dtype=torch.float32)
        frequencies = torch.outer(t, frequencies)
        rotations = torch.polar(torch.ones_like(frequencies), frequencies)

        self.register_buffer("rotations", rotations, persistent=False)
