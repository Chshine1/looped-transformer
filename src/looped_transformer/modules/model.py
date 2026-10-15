import math

import torch
import torch.nn as nn
from torch import Tensor

from looped_transformer.config.model import ModelOptions
from looped_transformer.modules.attn_res import AttnRes
from looped_transformer.modules.halting_cell import HaltingCell, HaltingResult
from looped_transformer.modules.masked_mean_pooling import MaskedMeanPooling
from looped_transformer.modules.swi_glu_transformer_block import SwiGLUTransformerBlock
from looped_transformer.types.model import ModelForwardOutput
from looped_transformer.types.tokenizer import TokenizerInfo
from looped_transformer.utils.shared_cache import PositionalRoPECache
from looped_transformer.utils.shared_cache import TimedMultiHeadRoPECache


class LoopedMNLI(nn.Module):
    class _MultiHeadRotation(nn.Module):
        _timed_rope_cache: TimedMultiHeadRoPECache

        def __init__(
            self,
            timed_rope_cache: TimedMultiHeadRoPECache,
            model_options: ModelOptions,
        ):
            super().__init__()
            self._model_options = model_options
            object.__setattr__(self, "_timed_rope_cache", timed_rope_cache)

        def forward(
            self,
            t: int,
            h: Tensor,
        ) -> Tensor:
            if t == 0:
                return h

            batch_size, seq_length, _ = h.shape
            h = h.reshape(
                batch_size,
                seq_length,
                self._model_options.transformer.attention_heads_count,
                -1,
            )
            rotation = (
                self._timed_rope_cache.rotations[t - 1]
                if t > 0
                else torch.conj(self._timed_rope_cache.rotations[-t - 1])
            )
            h1, h2 = h.chunk(2, dim=-1)
            complex_hidden_state = torch.complex(h1, h2) * rotation.unsqueeze(-1)
            return torch.cat(
                [complex_hidden_state.real, complex_hidden_state.imag], dim=-1
            ).reshape(batch_size, seq_length, -1)

    class _Loop(nn.Module):
        def __init__(
            self,
            positional_rope_cache: PositionalRoPECache,
            timed_rope_cache: TimedMultiHeadRoPECache,
            model_options: ModelOptions,
        ):
            super().__init__()
            self._model_options = model_options

            dim = model_options.hidden_state_dimension
            ffn_dim = model_options.transformer.swi_glu_ffn_dimension
            heads_count = model_options.transformer.attention_heads_count
            dropout = model_options.dropout

            self._transformer_block1 = SwiGLUTransformerBlock(
                positional_rope_cache, dim, ffn_dim, heads_count, dropout
            )
            self._transformer_block2 = SwiGLUTransformerBlock(
                positional_rope_cache, dim, ffn_dim, heads_count, dropout
            )
            self._loop_normalization = nn.RMSNorm(dim)
            self._rotation = LoopedMNLI._MultiHeadRotation(
                timed_rope_cache, model_options
            )

            self._transformer_scale = (1.0 / model_options.max_iterations) / math.sqrt(
                2
            )

        def forward(
            self,
            h: Tensor,
            padding_mask: Tensor,
        ) -> Tensor:
            h = self._transformer_block1(h, padding_mask, self._transformer_scale)
            h = self._transformer_block2(h, padding_mask, self._transformer_scale)
            h = self._loop_normalization(h)
            return self._rotation(1, h)

    def __init__(
        self,
        model_options: ModelOptions,
        tokenizer_info: TokenizerInfo,
    ):
        super().__init__()
        self._model_options = model_options

        dim = model_options.hidden_state_dimension
        dropout = model_options.dropout

        heads_count = model_options.transformer.attention_heads_count
        head_dimension = model_options.hidden_state_dimension // heads_count
        self._positional_rope = PositionalRoPECache(
            head_dimension, model_options.max_sequence_length
        )
        self._timed_rope = TimedMultiHeadRoPECache(
            heads_count, model_options.max_iterations, 4.8
        )

        self._token_embedding = nn.Embedding(
            tokenizer_info.vocab_size,
            dim,
            padding_idx=tokenizer_info.pad_token_id,
        )
        self._dropout = nn.Dropout(dropout)

        self._attn_res = AttnRes(
            self._positional_rope,
            dim,
            heads_count,
            model_options.max_iterations,
            dropout,
        )
        self._loop = self._Loop(self._positional_rope, self._timed_rope, model_options)

        self._alignment_rotation = self._MultiHeadRotation(
            self._timed_rope, model_options
        )
        self._mean_pooling = MaskedMeanPooling()

        self._halting_cell = HaltingCell(self._positional_rope, model_options)

        self._classify_normalization = nn.RMSNorm(dim)
        self._classifier = nn.Linear(dim, model_options.output_classes_count)

    def forward(self, input_ids: Tensor, padding_mask: Tensor) -> ModelForwardOutput:
        dimension = self._model_options.hidden_state_dimension
        batch_size, seq_length = input_ids.shape

        h0: Tensor = self._token_embedding(input_ids)
        h: Tensor = h0
        history_states: list[Tensor] = [h0]

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

        injection_scale = 1.0 / self._model_options.max_iterations

        for t in range(self._model_options.max_iterations):
            attn_res: Tensor = self._attn_res(h, history_states[:-1], padding_mask)
            h = h + injection_scale * attn_res
            h = self._loop(h, padding_mask)

            aligned = self._alignment_rotation(-(t + 1), h)
            pooled = self._mean_pooling(aligned, padding_mask)
            logits = self._classifier(self._classify_normalization(pooled))

            logits_list.append(logits)

            halting_result: HaltingResult = self._halting_cell(
                t, aligned.detach(), halting_hidden_state, padding_mask
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
