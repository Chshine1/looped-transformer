from torch import nn, Tensor
from torch.nn import functional


class SwiGLUBasedFFN(nn.Module):
    def __init__(self, model_dimension: int, ffn_dimension: int, dropout: float = 0.1):
        super().__init__()
        self._dropout = nn.Dropout(dropout)

        self._gate_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)
        self._up_transform   = nn.Linear(model_dimension, ffn_dimension, bias=False)
        self._down_transform = nn.Linear(ffn_dimension, model_dimension, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = functional.silu(self._gate_transform(x))
        up = self._up_transform(x)
        hidden = self._dropout(gate * up)
        return self._down_transform(hidden)