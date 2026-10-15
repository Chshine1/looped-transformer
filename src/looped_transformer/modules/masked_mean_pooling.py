from torch import Tensor
from torch import nn


class MaskedMeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, padding_mask: Tensor) -> Tensor:
        mask = padding_mask.unsqueeze(-1).to(x.dtype)
        return (x * mask).sum(1) / mask.sum(1).clamp(min=1)
