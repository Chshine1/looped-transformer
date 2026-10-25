from typing import cast

from torch import Tensor, nn

from looped_transformer.core.config.base import (
    ExclusiveProbability,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.core.nn.masked_mean_pooling import MaskedMeanPooling
from looped_transformer.core.nn.multi_head_attention import (
    AttentionOptions,
    RoPEMultiHeadAttention,
)
from looped_transformer.core.nn.positional_rope import (
    PositionalRoPECache,
    PositionalRoPEOptions,
)


class ControllerTeacherOptions(OptionsModel):
    model_dimension: PositiveInt
    ffn_dimension: PositiveInt
    heads_count: PositiveInt
    dropout: ExclusiveProbability


class ControllerTeacher(nn.Module):
    def __init__(self, options: ControllerTeacherOptions, sequence: SequenceOptions):
        super().__init__()
        dropout = options.dropout
        dim = options.model_dimension
        heads_count = options.heads_count

        self._positional_rope_cache = PositionalRoPECache(
            PositionalRoPEOptions(
                head_dimension=dim // heads_count,
                max_sequence_length=sequence.max_sequence_length,
            )
        )
        self._gru = nn.GRU(
            input_size=dim,
            hidden_size=dim // 2,
            num_layers=1,
            batch_first=False,
            bidirectional=True,
        )

        self._attention_normalization = nn.RMSNorm(dim)
        self._attention = RoPEMultiHeadAttention(
            self._positional_rope_cache,
            AttentionOptions(
                model_dimension=dim,
                heads_count=heads_count,
                dropout=dropout,
            ),
        )
        self._attention_dropout = nn.Dropout(dropout)

        mlp_dim = options.ffn_dimension
        self._classifier_normalization = nn.RMSNorm(dim)
        self._masked_mean_pooling = MaskedMeanPooling()
        self._classifier = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, 1),
        )

    def forward(self, aligned_hidden_states: Tensor, padding_mask: Tensor) -> Tensor:
        t, batch_size, seq_length, dimension = aligned_hidden_states.shape

        flattened_hidden = aligned_hidden_states.reshape(t, -1, dimension)
        gru_output, _ = cast(tuple[Tensor, Tensor], self._gru(flattened_hidden))
        h = gru_output.reshape(
            t, batch_size, seq_length, dimension
        ) * padding_mask.unsqueeze(-1).to(gru_output.dtype)
        h = h.reshape(t * batch_size, -1, dimension)

        mask = padding_mask.repeat(t, 1)
        attention: Tensor = self._attention_normalization(h)
        attention = self._attention(attention, attention, mask)

        out: Tensor = self._classifier_normalization(
            h + self._attention_dropout(attention)
        )
        out = self._masked_mean_pooling(out, mask).reshape(t, batch_size, dimension)
        out = self._classifier(out)
        return out.squeeze(-1)
