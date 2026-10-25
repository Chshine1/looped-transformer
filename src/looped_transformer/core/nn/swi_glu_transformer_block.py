"""Reusable SwiGLU transformer block."""

from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.core.config.base import (
    ExclusiveProbability,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.core.nn.batch_dropout import BatchDropout, BatchDropoutOptions
from looped_transformer.core.nn.multi_head_attention import (
    AttentionOptions,
    RoPEMultiHeadAttention,
)
from looped_transformer.core.nn.positional_rope import PositionalRoPECache
from looped_transformer.core.nn.residual_scale import ResidualScale


class TransformerBlockOptions(OptionsModel):
    model_dimension: PositiveInt
    ffn_dimension: PositiveInt
    heads_count: PositiveInt
    attention_dropout: ExclusiveProbability
    residual_dropout: ExclusiveProbability
    ffn_dropout: ExclusiveProbability
    stochastic_depth: ExclusiveProbability


class SwiGLUTransformerBlock(nn.Module):
    def __init__(
        self,
        positional_rope_cache: PositionalRoPECache,
        options: TransformerBlockOptions,
    ):
        super().__init__()
        model_dimension = options.model_dimension
        ffn_dimension = options.ffn_dimension

        self._normalization1 = nn.RMSNorm(model_dimension)

        self._transformer_attention = RoPEMultiHeadAttention(
            positional_rope_cache,
            AttentionOptions(
                model_dimension=model_dimension,
                heads_count=options.heads_count,
                dropout=options.attention_dropout,
            ),
        )

        self._dropout1 = nn.Dropout(options.residual_dropout)
        batch_dropout_options = BatchDropoutOptions(
            probability=options.stochastic_depth
        )
        self._batch_dropout1 = BatchDropout(batch_dropout_options)

        self._normalization2 = nn.RMSNorm(model_dimension)

        self._gate_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)
        self._up_transform = nn.Linear(model_dimension, ffn_dimension, bias=False)

        self._swi_glu_dropout = nn.Dropout(options.ffn_dropout)
        self._down_transform = nn.Linear(ffn_dimension, model_dimension, bias=False)

        self._dropout2 = nn.Dropout(options.residual_dropout)
        self._batch_dropout2 = BatchDropout(batch_dropout_options)
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

        return hidden_state + self._batch_dropout2(
            residue_scale * self._ffn_residual_scale() * self._dropout2(ffn)
        )
