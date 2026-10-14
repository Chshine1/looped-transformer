import math
import torch
import torch.nn as nn
from torch import Tensor

from looped_transformer.config.model import ModelOptions
from looped_transformer.modules.halting_cell import HaltingCell, HaltingResult
from looped_transformer.modules.multi_head_attention import RoPEMultiHeadAttention
from looped_transformer.modules.swi_glu_transformer_block import SwiGLUTransformerBlock
from looped_transformer.types.model import ModelForwardOutput
from looped_transformer.types.tokenizer import TokenizerInfo


class LoopedMNLI(nn.Module):
    UNIQUE_TRANSFORMER_BLOCKS = 2

    class _Loop(nn.Module):
        def __init__(self, model_options: ModelOptions):
            super().__init__()

            dim = model_options.hidden_state_dimension
            ffn_dim = model_options.transformer.swi_glu_ffn_dimension
            heads_count = model_options.transformer.attention_heads_count
            max_seq_length = model_options.max_sequence_length
            dropout = model_options.dropout

            self._input_injection = RoPEMultiHeadAttention(
                dim, heads_count, max_seq_length, dropout
            )
            self._transformer_block1 = SwiGLUTransformerBlock(
                dim, ffn_dim, heads_count, max_seq_length, dropout
            )
            self._transformer_block2 = SwiGLUTransformerBlock(
                dim, ffn_dim, heads_count, max_seq_length, dropout
            )
            self._loop_normalization = nn.RMSNorm(dim)

            self._injection_scale = 1.0 / model_options.max_iterations
            self._transformer_scale = self._injection_scale / math.sqrt(2)

        def forward(
            self,
            t: int,
            h0: Tensor,
            h: Tensor,
            padding_mask: Tensor,
            cos: Tensor,
            sin: Tensor,
        ) -> Tensor:
            if t > 0:
                h = h + self._injection_scale * self._input_injection(
                    h, h0, padding_mask
                )

            h = self._transformer_block1(h, padding_mask, self._transformer_scale)
            h = self._transformer_block2(h, padding_mask, self._transformer_scale)
            h = self._loop_normalization(h)

            h1, h2 = h.chunk(2, dim=-1)
            return torch.cat(
                [h1 * cos[0] - h2 * sin[0], h2 * cos[0] + h1 * sin[0]], dim=-1
            )

    def __init__(
        self,
        model_options: ModelOptions,
        tokenizer_info: TokenizerInfo,
    ):
        super().__init__()
        self._model_options = model_options

        dim = model_options.hidden_state_dimension
        dropout = model_options.dropout

        loop_rope_cos, loop_rope_sin = self._compute_rope_transformations(
            dim, model_options.max_iterations
        )
        self.register_buffer("_loop_rope_cos", loop_rope_cos, persistent=False)
        self.register_buffer("_loop_rope_sin", loop_rope_sin, persistent=False)

        self._token_embedding = nn.Embedding(
            tokenizer_info.vocab_size,
            dim,
            padding_idx=tokenizer_info.pad_token_id,
        )
        self._dropout = nn.Dropout(dropout)

        self._loop = self._Loop(model_options)

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

        cos = self._loop_rope_cos.to(dtype=h.dtype)
        sin = self._loop_rope_sin.to(dtype=h.dtype)

        logits_list: list[Tensor] = []
        hazards_list: list[Tensor] = []
        survivals_list: list[Tensor] = [
            torch.ones(batch_size, device=h.device, dtype=h.dtype)
        ]

        halting_hidden_state = torch.zeros(
            batch_size, seq_length, dimension, device=h.device, dtype=h.dtype
        )
        halting_times = (self._model_options.max_iterations - 1) * torch.ones(
            batch_size, device=h.device, dtype=torch.long
        )
        batches_halted = torch.zeros(batch_size, device=h.device, dtype=torch.bool)

        for t in range(self._model_options.max_iterations):
            h = self._loop(t, h0, h, padding_mask, cos, sin)

            h1, h2 = h.chunk(2, dim=-1)
            aligned_hidden_state = torch.cat(
                [h1 * cos[t] + h2 * sin[t], h2 * cos[t] - h1 * sin[t]], dim=-1
            )

            mask = padding_mask.unsqueeze(-1).to(aligned_hidden_state.dtype)
            pooled = (aligned_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
            logits = self._classifier(self._classify_normalization(pooled))

            logits_list.append(logits)

            halting_result: HaltingResult = self._halting_cell(
                t, aligned_hidden_state.detach(), halting_hidden_state, padding_mask
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

        batch_indices = torch.arange(batch_size, device=h.device)

        return {
            "mode": "eval",
            "logits": timed_logits[halting_times, batch_indices],
        }
