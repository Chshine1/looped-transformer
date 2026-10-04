import math
from typing import Literal, TypedDict

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional

from looped_transformer.multi_head_attention import MultiHeadAttention


class TrainOutput(TypedDict):
    mode: Literal["train"]
    timed_logits: Tensor
    halt_probabilities: Tensor


class EvalOutput(TypedDict):
    mode: Literal["eval"]
    logits: Tensor


ForwardOutput = TrainOutput | EvalOutput


class LoopedMNLI(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes,
        max_iter: int,
        d=128,
        n_head=4,
        ff=256,
        max_len=128,
        dropout=0.1,
        gamma_init=0.95,
        break_adjust_scale: float = 0.9,
        break_hazard_threshold: float = 0.6,
        break_survival_threshold: float = 0.05,
    ):
        super().__init__()
        self._d_model = d
        self._n_classes = num_classes

        self._max_iter = max_iter
        self._time_encoding = nn.Embedding(max_iter, d)

        self._log_gamma_logit = nn.Parameter(
            torch.tensor(math.log(gamma_init / (1 - gamma_init)))
        )
        self._loop_normalization = nn.RMSNorm(d)

        self._token_embedding = nn.Embedding(vocab_size, d, padding_idx=0)
        self._pos_encoding = nn.Embedding(max_len, d)
        self._dropout = nn.Dropout(dropout)

        self._input_injection = MultiHeadAttention(d, n_head, dropout)
        self._injection_gate = nn.Parameter(torch.zeros(d))

        self._normalization1 = nn.RMSNorm(d)
        self._transformer_attention = MultiHeadAttention(d, n_head, dropout)
        self._dropout1 = nn.Dropout(dropout)

        self._normalization2 = nn.RMSNorm(d)
        self._transformer_ffn = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff, d),
        )
        self._dropout2 = nn.Dropout(dropout)

        self._break_hazard_threshold = break_hazard_threshold
        self._break_survival_threshold = break_survival_threshold

        self._break_adjust_scale = break_adjust_scale
        self._break_baseline_alpha = nn.Parameter(torch.tensor(-2.0))
        self._break_baseline_beta = nn.Parameter(torch.tensor(-2.0))

        self._break_state_rnn_cell = nn.GRUCell(
            input_size=self._d_model, hidden_size=self._d_model
        )
        self._normalization3 = nn.RMSNorm(d)
        self._break_attention = MultiHeadAttention(d, n_head, dropout)
        self._dropout3 = nn.Dropout(dropout)

        self._normalization4 = nn.RMSNorm(d)

        break_adjust_mlp_last = nn.Linear(ff, 1)
        nn.init.zeros_(break_adjust_mlp_last.weight)
        nn.init.constant_(break_adjust_mlp_last.bias, -2.0)
        self._break_adjust_mlp = nn.Sequential(
            nn.Linear(d, ff),
            nn.GELU(),
            nn.Dropout(dropout),
            break_adjust_mlp_last,
        )

        self._classify_normalization = nn.RMSNorm(d)
        self._classifier = nn.Linear(d, num_classes)

    def forward(self, input_ids: Tensor, padding_mask: Tensor) -> ForwardOutput:
        batch_size, seq_length = input_ids.shape

        pos_indices = torch.arange(seq_length, device=input_ids.device)
        h0: Tensor = self._dropout(
            self._token_embedding(input_ids)
            + self._pos_encoding(pos_indices)[None, :, :]
        )

        t_indices = torch.arange(
            self._max_iter, device=input_ids.device, dtype=torch.long
        )
        time_encodings = self._time_encoding(t_indices)

        h = h0
        break_hidden_state = torch.zeros(
            batch_size, seq_length, self._d_model, device=input_ids.device
        )

        logits_list: list[Tensor] = []
        survival_list: list[Tensor] = [torch.ones(batch_size, device=input_ids.device)]
        hazard_list: list[Tensor] = []

        break_times = (self._max_iter - 1) * torch.ones(
            batch_size, device=input_ids.device, dtype=torch.long
        )
        batches_halted = torch.zeros(batch_size, device=input_ids.device).bool()

        for t in range(self._max_iter):
            scale = torch.exp(t * functional.logsigmoid(self._log_gamma_logit))

            h = (
                h
                + self._injection_gate * self._input_injection(h, h0, padding_mask)
                + time_encodings[t][None, None, :]
            )

            attention: Tensor = self._normalization1(h)
            attention = self._transformer_attention(attention, attention, padding_mask)
            h = h + scale * self._dropout1(attention)

            ffn: Tensor = self._normalization2(h)
            ffn = self._transformer_ffn(ffn)
            h = h + scale * self._dropout2(ffn)

            h = self._loop_normalization(h)

            mask = padding_mask.unsqueeze(-1).float()
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1)
            logits = self._classifier(self._classify_normalization(pooled))

            logits_list.append(logits)

            break_dep_state = attention.detach()
            break_hidden_new: Tensor = self._break_state_rnn_cell(
                break_dep_state.reshape(batch_size * seq_length, self._d_model),
                break_hidden_state.reshape(batch_size * seq_length, self._d_model),
            )
            validity_mask = padding_mask.reshape(batch_size * seq_length, 1).to(
                break_hidden_new.dtype
            )
            break_hidden_state = (validity_mask * break_hidden_new).reshape(
                batch_size, seq_length, self._d_model
            )

            break_attention: Tensor = self._normalization3(break_hidden_state)
            break_attention = self._break_attention(
                break_attention, break_attention, padding_mask
            )
            out: Tensor = self._normalization4(
                break_hidden_state + self._dropout3(break_attention)
            )
            out = self._break_adjust_mlp(out)

            hazard_logit = (
                self._break_baseline_alpha
                + functional.softplus(self._break_baseline_beta) * t
                + torch.tanh(out) * self._break_adjust_scale
            )
            hazard_logit = (hazard_logit * mask).sum(1) / mask.sum(1).clamp(min=1)
            hazard_logit = hazard_logit.squeeze(-1)

            hazard = torch.sigmoid(hazard_logit)
            hazard_list.append(hazard_logit)
            survival = survival_list[-1] * (1 - hazard)
            survival_list.append(survival)

            if not self.training:
                continue

            halt_batches = (~batches_halted) & (
                (hazard > self._break_hazard_threshold)
                | (survival < self._break_survival_threshold)
            )
            batches_halted = batches_halted | halt_batches
            break_times = break_times - (self._max_iter - 1 - t) * halt_batches

            if torch.all(batches_halted):
                break

        timed_logits = torch.stack(logits_list)

        if self.training:
            hazards = torch.stack(hazard_list)
            hazards[-1, :] = 1.0
            survivals = torch.stack(survival_list[:-1])

            return {
                "mode": "train",
                "timed_logits": timed_logits,
                "halt_probabilities": hazards * survivals,
            }

        batch_indices = torch.arange(batch_size, device=input_ids.device)

        return {
            "mode": "eval",
            "logits": timed_logits[break_times, batch_indices],
        }
