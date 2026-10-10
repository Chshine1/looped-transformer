import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional

from looped_transformer.config.model import ModelOptions
from looped_transformer.config.tokenizer import TokenizerOptions
from looped_transformer.modules.halting_cell import HaltingCell, HaltingResult
from looped_transformer.modules.multi_head_attention import RoPEMultiHeadAttention
from looped_transformer.modules.swi_glu_transformer_block import SwiGLUTransformerBlock
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
        ffn_dim = model_options.transformer.swi_glu_ffn_dimension
        if (
            tokenizer_options.vocab_size is None
            or tokenizer_options.pad_token_id is None
        ):
            raise ValueError(
                "Tokenizer vocabulary metadata must be resolved before model creation"
            )

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
        self._dropout = nn.Dropout(dropout)

        max_seq_length = model_options.max_sequence_length
        self._input_injection = RoPEMultiHeadAttention(
            dim, heads_count, max_seq_length, dropout
        )
        self._injection_gate = nn.Parameter(torch.zeros(dim))

        self._transformer_block1 = SwiGLUTransformerBlock(
            dim, ffn_dim, heads_count, max_seq_length, dropout
        )
        self._transformer_block2 = SwiGLUTransformerBlock(
            dim, ffn_dim, heads_count, max_seq_length, dropout
        )

        self._halting_cell = HaltingCell(model_options)

        self._classify_normalization = nn.RMSNorm(dim)
        self._classifier = nn.Linear(dim, model_options.output_classes_count)

    @staticmethod
    def _compute_rope_transformations(model_dimension: int, max_iterations: int):
        frequencies = 1.0 / (
            5.1 ** (torch.arange(0, model_dimension, 2).float() / model_dimension)
        )
        t = torch.arange(1, max_iterations + 1, dtype=torch.float32)
        frequencies = torch.outer(t, frequencies)
        cos = torch.cos(frequencies)
        sin = torch.sin(frequencies)
        return cos, sin

    def forward(self, input_ids: Tensor, padding_mask: Tensor) -> ModelForwardOutput:
        dimension = self._model_options.hidden_state_dimension
        batch_size, seq_length = input_ids.shape

        h0: Tensor = self._token_embedding(input_ids)
        h: Tensor = self._dropout(h0)

        cos, sin = self._compute_rope_transformations(
            dimension, self._model_options.max_iterations
        )

        logits_list: list[Tensor] = []
        hazards_list: list[Tensor] = []
        survivals_list: list[Tensor] = [
            torch.ones(batch_size, device=input_ids.device, dtype=h.dtype)
        ]

        halting_hidden_state = torch.zeros(
            batch_size, seq_length, dimension, device=input_ids.device, dtype=h.dtype
        )
        halting_times = (self._model_options.max_iterations - 1) * torch.ones(
            batch_size, device=input_ids.device, dtype=torch.long
        )
        batches_halted = torch.zeros(batch_size, device=input_ids.device).bool()

        for t in range(self._model_options.max_iterations):
            scale = torch.exp(t * functional.logsigmoid(self._log_gamma_logit))

            if t > 0:
                h = h + self._injection_gate * self._input_injection(
                    h, h0, padding_mask
                )

            h = self._transformer_block1(h, padding_mask, scale / 2)
            h = self._transformer_block2(h, padding_mask, scale / 2)
            h = self._loop_normalization(h)

            h1, h2 = h.chunk(2, dim=-1)
            h = torch.cat(
                [h1 * cos[0] - h2 * sin[0], h2 * cos[0] + h1 * sin[0]], dim=-1
            )

            output = torch.cat(
                [h1 * cos[t] + h2 * sin[t], h2 * cos[t] - h1 * sin[t]], dim=-1
            )

            mask = padding_mask.unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
            logits = self._classifier(self._classify_normalization(pooled))

            logits_list.append(logits)

            halting_result: HaltingResult = self._halting_cell(
                t, output.detach(), halting_hidden_state, padding_mask
            )
            halting_hidden_state = halting_result["hidden_state"]

            hazards = halting_result["hazards"]
            hazards_list.append(hazards)
            survivals = survivals_list[-1] * (1 - hazards)
            survivals_list.append(survivals)

            if self.training:
                continue

            halt_batches = (~batches_halted) & (
                (hazards > self._model_options.evaluation.halt_hazard_threshold)
                | (survivals < self._model_options.evaluation.halt_survival_threshold)
            )
            batches_halted = batches_halted | halt_batches
            halting_times = (
                halting_times
                - (self._model_options.max_iterations - 1 - t) * halt_batches
            )

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
