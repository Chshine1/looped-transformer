import torch
from torch import Tensor, nn

from looped_transformer.config.base import ExclusiveProbability, OptionsModel


class BatchDropoutOptions(OptionsModel):
    probability: ExclusiveProbability = 0.0


class BatchDropout(nn.Module):
    def __init__(self, options: BatchDropoutOptions):
        super().__init__()
        self._probability = options.probability

    def forward(self, value: Tensor) -> Tensor:
        if not self.training or self._probability == 0:
            return value
        keep_probability = 1 - self._probability
        mask_shape = (value.size(0),) + (1,) * (value.ndim - 1)
        mask = torch.empty(mask_shape, device=value.device, dtype=value.dtype)
        mask.bernoulli_(keep_probability)
        return value * mask / keep_probability
