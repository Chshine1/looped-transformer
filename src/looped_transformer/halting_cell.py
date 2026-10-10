import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.config.model import ModelOptions
from looped_transformer.multi_head_attention import RoPEMultiHeadAttention
from looped_transformer.types.halting_cell import HaltingResult


class HaltingCell(nn.Module):
    def __init__(self, model_options: ModelOptions):
        super().__init__()

        dim = model_options.hidden_state_dimension
        dropout = model_options.training.dropout
        heads_count = model_options.transformer.attention_heads_count
        mlp_dim = model_options.transformer.gelu_ffn_dimension

        self._hazard_adjustment_scale = nn.Parameter(
            torch.tensor(model_options.training.halt_hazard_adjustment_scale)
        )
        self._hazard_baseline_offset = nn.Parameter(
            torch.tensor(model_options.training.halt_hazard_baseline_offset)
        )
        self._hazard_baseline_scale = nn.Parameter(
            torch.tensor(model_options.training.halt_hazard_baseline_scale)
        )

        self._rnn_cell = nn.GRUCell(input_size=dim, hidden_size=dim)
        self._attention_normalization = nn.RMSNorm(dim)
        self._attention = RoPEMultiHeadAttention(
            dim, heads_count, model_options.max_sequence_length, dropout
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
        mask = padding_mask.unsqueeze(-1).to(hazards_logit.dtype)
        hazards_logit = (hazards_logit * mask).sum(1) / mask.sum(1).clamp(min=1)
        hazards_logit = hazards_logit.squeeze(-1)

        return {
            "hidden_state": hidden_state,
            "hazards": torch.sigmoid(hazards_logit),
        }
