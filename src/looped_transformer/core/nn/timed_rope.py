"""Cached loop-time rotary embeddings."""

import torch
from torch import Tensor, nn

from looped_transformer.core.config.base import OptionsModel, PositiveFloat, PositiveInt


class TimedRoPEOptions(OptionsModel):
    heads_count: PositiveInt
    max_iterations: PositiveInt
    base_frequency: PositiveFloat


class TimedMultiHeadRoPECache(nn.Module):
    cosine: Tensor
    sine: Tensor

    def __init__(self, options: TimedRoPEOptions):
        super().__init__()
        frequencies = 1.0 / (
            options.base_frequency
            ** (torch.arange(0, options.heads_count).float() / options.heads_count)
        )
        times = torch.arange(1, options.max_iterations + 1, dtype=torch.float32)
        frequencies = torch.outer(times, frequencies)
        self.register_buffer("cosine", frequencies.cos(), persistent=False)
        self.register_buffer("sine", frequencies.sin(), persistent=False)
