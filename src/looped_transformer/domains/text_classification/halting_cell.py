import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.core.abstractions.halting import HaltingResult
from looped_transformer.core.config.base import (
    ExclusiveProbability,
    FiniteFloat,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.core.modules.nn.masked_mean_pooling import MaskedMeanPooling
from looped_transformer.core.modules.nn.multi_head_attention import (
    AttentionOptions,
    RoPEMultiHeadAttention,
)
from looped_transformer.core.modules.nn.positional_rope import PositionalRoPECache


class HaltingCellOptions(OptionsModel):
    model_dimension: PositiveInt
    ffn_dimension: PositiveInt
    heads_count: PositiveInt
    dropout: ExclusiveProbability
    hazard_adjustment_scale: FiniteFloat
    hazard_baseline_offset: FiniteFloat
    hazard_baseline_scale: FiniteFloat


class HaltingCell(nn.Module):
    def __init__(
        self, positional_rope_cache: PositionalRoPECache, options: HaltingCellOptions
    ):
        super().__init__()

        dim = options.model_dimension
        dropout = options.dropout
        heads_count = options.heads_count
        mlp_dim = options.ffn_dimension

        self._hazard_adjustment_scale = nn.Parameter(
            torch.tensor(options.hazard_adjustment_scale)
        )
        self._hazard_baseline_offset = nn.Parameter(
            torch.tensor(options.hazard_baseline_offset)
        )
        self._hazard_baseline_scale = nn.Parameter(
            torch.tensor(options.hazard_baseline_scale)
        )

        self._rnn_cell = nn.GRUCell(input_size=dim, hidden_size=dim)
        self._attention_normalization = nn.RMSNorm(dim)
        self._attention = RoPEMultiHeadAttention(
            positional_rope_cache,
            AttentionOptions(
                model_dimension=dim,
                heads_count=heads_count,
                dropout=dropout,
            ),
        )
        self._attention_dropout = nn.Dropout(dropout)

        self._mlp_normalization = nn.RMSNorm(dim)

        adjustment_mlp_last = nn.Linear(mlp_dim, 1)
        nn.init.zeros_(adjustment_mlp_last.weight)
        nn.init.constant_(adjustment_mlp_last.bias, -2.0)

        self._adjustment_mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            adjustment_mlp_last,
        )

        self._masked_mean_pooling = MaskedMeanPooling()

    def forward(
        self,
        time: int,
        dependent_state: Tensor,
        hidden_state: Tensor,
        padding_mask: Tensor,
    ) -> HaltingResult:
        batch_size, seq_length, dimension = dependent_state.shape

        hidden_new: Tensor = self._rnn_cell(
            dependent_state.reshape(batch_size * seq_length, dimension),
            hidden_state.reshape(batch_size * seq_length, dimension),
        )
        validity_mask = padding_mask.reshape(batch_size * seq_length, 1).to(
            hidden_new.dtype
        )
        hidden_state = (validity_mask * hidden_new).reshape(
            batch_size, seq_length, dimension
        )

        attention: Tensor = self._attention_normalization(hidden_state)
        attention = self._attention(attention, attention, padding_mask)
        out: Tensor = self._mlp_normalization(
            hidden_state + self._attention_dropout(attention)
        )
        out = self._adjustment_mlp(out)

        hazards_logit = (
            self._hazard_baseline_offset
            + functional.softplus(self._hazard_baseline_scale) * time
            + torch.tanh(out) * self._hazard_adjustment_scale
        )
        hazards_logit = self._masked_mean_pooling(hazards_logit, padding_mask)
        hazards_logit = hazards_logit.squeeze(-1)

        return {
            "hidden_state": hidden_state,
            "hazards": torch.sigmoid(hazards_logit),
        }
