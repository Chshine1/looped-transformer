from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.modules.batch_dropout import BatchDropout
from looped_transformer.modules.multi_head_attention import RoPEMultiHeadAttention
from looped_transformer.modules.residual_scale import ResidualScale
from looped_transformer.utils.shared_cache import PositionalRoPECache


class SwiGLUTransformerBlock(nn.Module):
    def __init__(
        self,
        positional_rope_cache: PositionalRoPECache,
        model_dimension: int,
        ffn_dimension: int,
        heads_count: int,
        attention_dropout: float,
        residual_dropout: float,
        ffn_dropout: float,
        stochastic_depth: float,
    ):
        super().__init__()

        self._normalization1 = nn.RMSNorm(model_dimension)

        self._transformer_attention = RoPEMultiHeadAttention(
            positional_rope_cache, model_dimension, heads_count, attention_dropout
        )

        self._dropout1 = nn.Dropout(residual_dropout)
        self._batch_dropout1 = BatchDropout(stochastic_depth)

        self._normalization2 = nn.RMSNorm(model_dimension)

        self._gate_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)
        self._up_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)

        self._swi_glu_dropout = nn.Dropout(ffn_dropout)
        self._down_transform = nn.Linear(ffn_dimension, model_dimension, bias=False)

        self._dropout2 = nn.Dropout(residual_dropout)
        self._batch_dropout1 = BatchDropout(stochastic_depth)
        self._attention_residual_scale = ResidualScale()
        self._ffn_residual_scale = ResidualScale()

    def forward(
        self, hidden_state: Tensor, padding_mask: Tensor, residue_scale: float
    ) -> Tensor:
        attention: Tensor = self._normalization1(hidden_state)
        attention = self._transformer_attention(attention, attention, padding_mask)
        hidden_state = hidden_state + self._batch_dropout1(
            residue_scale * self._attention_residual_scale() * self._dropout1(attention)
        )

        ffn: Tensor = self._normalization2(hidden_state)

        gate = functional.silu(self._gate_transform(ffn))
        up = self._up_transform(hidden_state)
        ffn = self._down_transform(self._swi_glu_dropout(gate * up))

        return hidden_state + self._batch_dropout1(
            residue_scale * self._ffn_residual_scale() * self._dropout2(ffn)
        )
