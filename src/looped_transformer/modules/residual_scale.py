import torch
from torch import Tensor, nn


class ResidualScale(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.zeros(()))

    def forward(self) -> Tensor:
        return 2.0 * torch.sigmoid(self.logit)
