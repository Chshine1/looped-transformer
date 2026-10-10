from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.multi_head_attention import RoPEMultiHeadAttention


class SwiGLUTransformerBlock(nn.Module):
    def __init__(
        self,
        model_dimension: int,
        ffn_dimension: int,
        heads_count: int,
        max_sequence_length: int,
        dropout: float,
    ):
        super().__init__()

        self._normalization1 = nn.RMSNorm(model_dimension)
        self._transformer_attention = RoPEMultiHeadAttention(
            model_dimension, heads_count, max_sequence_length, dropout
        )
        self._dropout1 = nn.Dropout(dropout)

        self._normalization2 = nn.RMSNorm(model_dimension)

        self._gate_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)
        self._up_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)

        self._swi_glu_dropout = nn.Dropout(dropout)
        self._down_transform = nn.Linear(ffn_dimension, model_dimension, bias=False)

        self._dropout2 = nn.Dropout(dropout)

    def forward(
        self, hidden_state: Tensor, padding_mask: Tensor, residue_scale: float
    ) -> Tensor:
        attention: Tensor = self._normalization1(hidden_state)
        attention = self._transformer_attention(attention, attention, padding_mask)
        hidden_state = hidden_state + residue_scale * self._dropout1(attention)

        ffn: Tensor = self._normalization2(hidden_state)

        gate = functional.silu(self._gate_transform(ffn))
        up = self._up_transform(hidden_state)
        ffn = self._down_transform(self._swi_glu_dropout(gate * up))

        return hidden_state + residue_scale * self._dropout2(ffn)
