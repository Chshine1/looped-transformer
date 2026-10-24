import torch
from torch import Tensor, nn

from looped_transformer.config.base import OptionsModel, PositiveInt


class PositionalRoPEOptions(OptionsModel):
    head_dimension: PositiveInt
    max_sequence_length: PositiveInt


class PositionalRoPECache(nn.Module):
    cosine: Tensor
    sine: Tensor

    def __init__(self, options: PositionalRoPEOptions):
        super().__init__()
        frequencies = 1.0 / (
            100.0
            ** (
                torch.arange(0, options.head_dimension, 2).float()
                / options.head_dimension
            )
        )
        positions = torch.arange(options.max_sequence_length, dtype=torch.float32)
        frequencies = torch.outer(positions, frequencies)
        self.register_buffer("cosine", frequencies.cos(), persistent=False)
        self.register_buffer("sine", frequencies.sin(), persistent=False)
