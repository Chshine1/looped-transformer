import torch
from torch import Tensor, nn


class BatchDropout(nn.Module):
    def __init__(self, probability: float = 0.0):
        super().__init__()
        if not 0 <= probability < 1:
            raise ValueError("drop-path probability must be in [0, 1)")
        self._probability = probability

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self._probability == 0:
            return value
        keep_probability = 1 - self._probability
        mask_shape = (value.size(0),) + (1,) * (value.ndim - 1)
        mask = torch.empty(mask_shape, device=value.device, dtype=value.dtype)
        mask.bernoulli_(keep_probability)
        return value * mask / keep_probability
