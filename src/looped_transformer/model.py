import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional

from looped_transformer.config.model import ModelOptions
from looped_transformer.config.tokenizer import TokenizerOptions
from looped_transformer.halting_cell import HaltingCell, HaltingResult
from looped_transformer.multi_head_attention import MultiHeadAttention
from looped_transformer.types.model import ModelForwardOutput


class LoopedMNLI(nn.Module):
    def __init__(
        self,
        model_options: ModelOptions,
        tokenizer_options: TokenizerOptions,
    ):
        super().__init__()
        self._model_options = model_options

        dim = model_options.hidden_state_dimension
        dropout = model_options.training.dropout
        heads_count = model_options.transformer.attention_heads_count
        ffn_dim = model_options.transformer.ffn_dimension

        self._time_encoding = nn.Embedding(model_options.max_iterations, dim)

        self._log_gamma_logit = nn.Parameter(
            torch.tensor(
                math.log(
                    model_options.training.residue_time_scaling
                    / (1 - model_options.training.residue_time_scaling)
                )
            )
        )
        self._loop_normalization = nn.RMSNorm(dim)

        self._token_embedding = nn.Embedding(
            tokenizer_options.vocab_size,
            dim,
            padding_idx=tokenizer_options.pad_token_id,
        )
        self._pos_encoding = nn.Embedding(model_options.max_sequence_length, dim)
        self._dropout = nn.Dropout(dropout)

        self._input_injection = MultiHeadAttention(dim, heads_count, dropout)
        self._injection_gate = nn.Parameter(torch.zeros(dim))

        self._normalization1 = nn.RMSNorm(dim)
        self._transformer_attention = MultiHeadAttention(dim, heads_count, dropout)
        self._dropout1 = nn.Dropout(dropout)

        self._normalization2 = nn.RMSNorm(dim)
        self._transformer_ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
        )
        self._dropout2 = nn.Dropout(dropout)

        self._halting_cell = HaltingCell(model_options)

        self._classify_normalization = nn.RMSNorm(dim)
        self._classifier = nn.Linear(dim, model_options.output_classes_count)

    def forward(self, input_ids: Tensor, padding_mask: Tensor) -> ModelForwardOutput:
        dimension = self._model_options.hidden_state_dimension
        batch_size, seq_length = input_ids.shape

        pos_indices = torch.arange(seq_length, device=input_ids.device)
        h0: Tensor = self._dropout(
            self._token_embedding(input_ids)
            + self._pos_encoding(pos_indices)[None, :, :]
        )

        t_indices = torch.arange(
            self._model_options.max_iterations,
            device=input_ids.device,
            dtype=torch.long,
        )
        time_encodings = self._time_encoding(t_indices)

        h = h0

        logits_list: list[Tensor] = []
        hazards_list: list[Tensor] = []
        survivals_list: list[Tensor] = [torch.ones(batch_size, device=input_ids.device)]

        halting_hidden_state = torch.zeros(
            batch_size, seq_length, dimension, device=input_ids.device
        )
        halting_times = (self._max_iter - 1) * torch.ones(
            batch_size, device=input_ids.device, dtype=torch.long
        )
        batches_halted = torch.zeros(batch_size, device=input_ids.device).bool()

        for t in range(self._model_options.max_iterations):
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

            halting_result: HaltingResult = self._halting_cell(
                t, attention.detach(), halting_hidden_state, padding_mask
            )
            halting_hidden_state = halting_result["hidden_state"]

            hazards = halting_result["hazards"]
            hazards_list.append(hazards)
            survivals = survivals_list[-1] * (1 - hazards)
            survivals_list.append(survivals)

            if not self.training:
                continue

            halt_batches = (~batches_halted) & (
                (hazards > self._model_options.evaluation.halt_hazard_threshold)
                | (survivals < self._model_options.evaluation.halt_survival_threshold)
            )
            batches_halted = batches_halted | halt_batches
            halting_times = halting_times - (self._max_iter - 1 - t) * halt_batches

            if torch.all(batches_halted):
                break

        timed_logits = torch.stack(logits_list)

        if self.training:
            timed_hazards = torch.stack(hazards_list)
            timed_hazards[-1, :] = 1.0
            timed_survivals = torch.stack(survivals_list[:-1])

            return {
                "mode": "train",
                "timed_logits": timed_logits,
                "halt_probabilities": timed_hazards * timed_survivals,
            }

        batch_indices = torch.arange(batch_size, device=input_ids.device)

        return {
            "mode": "eval",
            "logits": timed_logits[halting_times, batch_indices],
        }
