from looped_transformer.domains.mnli.config.options import LoopOptions
import math
from typing import Self

import torch
import torch.nn as nn
from pydantic import model_validator
from torch import Tensor

from looped_transformer.core.abstractions.typed_module import IModule
from looped_transformer.core.config.base import ExclusiveProbability
from looped_transformer.core.config.base import FiniteFloat
from looped_transformer.core.config.base import OptionsModel, PositiveInt
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.core.modules.attention_residual import (
    AttentionResidualOptions,
    AttnRes,
)
from looped_transformer.core.modules.masked_mean_pooling import MaskedMeanPooling
from looped_transformer.core.modules.positional_rope import (
    PositionalRoPECache,
    PositionalRoPEOptions,
)
from looped_transformer.core.modules.swi_glu_transformer_block import (
    SwiGLUTransformerBlock,
    TransformerBlockOptions,
)
from looped_transformer.core.modules.timed_rope import (
    TimedMultiHeadRoPECache,
    TimedRoPEOptions,
)
from looped_transformer.domains.mnli.modules.halting_cell import (
    HaltingCell,
    HaltingCellOptions,
    HaltingResult,
)
from looped_transformer.domains.mnli.types.model import ModelForwardOutput
from looped_transformer.domains.mnli.types.tokenizer import TokenizerInfo
from looped_transformer.utils.torch.functional import apply_rotary


class ModelOptions(OptionsModel):
    class _DropoutOptions(OptionsModel):
        token: ExclusiveProbability = 0.0
        embedding: ExclusiveProbability = 0.1
        attention: ExclusiveProbability = 0.1
        residual: ExclusiveProbability = 0.1
        ffn: ExclusiveProbability = 0.1
        classifier: ExclusiveProbability = 0.1
        controller: ExclusiveProbability = 0.1
        stochastic_depth: ExclusiveProbability = 0.0

    class _TransformerOptions(OptionsModel):
        swi_glu_ffn_dimension: PositiveInt
        gelu_ffn_dimension: PositiveInt
        attention_heads_count: PositiveInt

    class _HaltingOptions(OptionsModel):
        hazard_baseline_offset: FiniteFloat = -2.0
        hazard_baseline_scale: FiniteFloat = -2.0
        hazard_adjustment_scale: FiniteFloat = 0.9

    hidden_state_dimension: PositiveInt
    output_classes_count: PositiveInt
    loop: LoopOptions
    dropout: _DropoutOptions
    transformer: _TransformerOptions
    halting: _HaltingOptions

    @model_validator(mode="after")
    def require_complete_attention_heads(self) -> Self:
        heads = self.transformer.attention_heads_count
        if self.hidden_state_dimension % heads:
            raise ValueError(
                "hidden_state_dimension must be divisible by attention_heads_count"
            )
        return self


class MultiHeadRotationOptions(OptionsModel):
    heads_count: PositiveInt


class LoopModuleOptions(OptionsModel):
    transformer_block: TransformerBlockOptions
    max_iterations: PositiveInt


class LoopedMNLI(nn.Module, IModule[Tensor, Tensor, ModelForwardOutput]):
    class _MultiHeadRotation(nn.Module):
        _timed_rope_cache: TimedMultiHeadRoPECache

        def __init__(
            self,
            timed_rope_cache: TimedMultiHeadRoPECache,
            options: MultiHeadRotationOptions,
        ):
            super().__init__()
            self._options = options
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
                self._options.heads_count,
                -1,
            )
            cosine = self._timed_rope_cache.cosine[abs(t) - 1]
            sine = self._timed_rope_cache.sine[abs(t) - 1]
            if t < 0:
                sine = -sine
            return apply_rotary(h, cosine.unsqueeze(-1), sine.unsqueeze(-1)).reshape(
                batch_size, seq_length, -1
            )

    class _Loop(nn.Module):
        def __init__(
            self,
            positional_rope_cache: PositionalRoPECache,
            timed_rope_cache: TimedMultiHeadRoPECache,
            options: LoopModuleOptions,
        ):
            super().__init__()
            block_options = options.transformer_block
            dim = block_options.model_dimension

            self._transformer_block1 = SwiGLUTransformerBlock(
                positional_rope_cache,
                block_options,
            )
            self._transformer_block2 = SwiGLUTransformerBlock(
                positional_rope_cache,
                block_options,
            )
            self._loop_normalization = nn.RMSNorm(dim)
            self._rotation = LoopedMNLI._MultiHeadRotation(
                timed_rope_cache,
                MultiHeadRotationOptions(heads_count=block_options.heads_count),
            )

            self._transformer_scale = (1.0 / options.max_iterations) / math.sqrt(2)

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
        sequence_options: SequenceOptions,
        tokenizer_info: TokenizerInfo,
    ):
        super().__init__()
        self._model_options = model_options

        dim = model_options.hidden_state_dimension
        dropout = model_options.dropout

        heads_count = model_options.transformer.attention_heads_count
        block_options = TransformerBlockOptions(
            model_dimension=dim,
            ffn_dimension=model_options.transformer.swi_glu_ffn_dimension,
            heads_count=heads_count,
            attention_dropout=dropout.attention,
            residual_dropout=dropout.residual,
            ffn_dropout=dropout.ffn,
            stochastic_depth=dropout.stochastic_depth,
        )
        head_dimension = model_options.hidden_state_dimension // heads_count
        self._positional_rope = PositionalRoPECache(
            PositionalRoPEOptions(
                head_dimension=head_dimension,
                max_sequence_length=sequence_options.max_sequence_length,
            )
        )
        self._timed_rope = TimedMultiHeadRoPECache(
            TimedRoPEOptions(
                heads_count=heads_count,
                max_iterations=model_options.loop.max_iterations,
                base_frequency=model_options.loop.time_rope_base_frequency,
            )
        )

        self._token_embedding = nn.Embedding(
            tokenizer_info.vocab_size,
            dim,
            padding_idx=tokenizer_info.pad_token_id,
        )
        self._token_dropout = dropout.token
        self._mask_token_id = tokenizer_info.mask_token_id
        self._special_token_ids = tokenizer_info.special_token_ids
        self._embedding_dropout = nn.Dropout(dropout.embedding)

        self._attn_res = AttnRes(
            self._positional_rope,
            AttentionResidualOptions(
                model_dimension=dim,
                heads_count=heads_count,
                max_iterations=model_options.loop.max_iterations,
                attention_dropout=dropout.attention,
                residual_dropout=dropout.residual,
                stochastic_depth=dropout.stochastic_depth,
            ),
        )
        self._loop = self._Loop(
            self._positional_rope,
            self._timed_rope,
            LoopModuleOptions(
                transformer_block=block_options,
                max_iterations=model_options.loop.max_iterations,
            ),
        )

        self._alignment_rotation = self._MultiHeadRotation(
            self._timed_rope, MultiHeadRotationOptions(heads_count=heads_count)
        )
        self._mean_pooling = MaskedMeanPooling()

        self._halting_cell = HaltingCell(
            self._positional_rope,
            HaltingCellOptions(
                model_dimension=dim,
                ffn_dimension=model_options.transformer.gelu_ffn_dimension,
                heads_count=heads_count,
                dropout=dropout.controller,
                hazard_adjustment_scale=model_options.halting.hazard_adjustment_scale,
                hazard_baseline_offset=model_options.halting.hazard_baseline_offset,
                hazard_baseline_scale=model_options.halting.hazard_baseline_scale,
            ),
        )

        self._classify_normalization = nn.RMSNorm(dim)
        self._classifier_dropout = nn.Dropout(dropout.classifier)
        self._classifier = nn.Linear(dim, model_options.output_classes_count)

    def forward(
        self,
        input_ids: Tensor,
        padding_mask: Tensor,
    ) -> ModelForwardOutput:
        dimension = self._model_options.hidden_state_dimension
        batch_size, seq_length = input_ids.shape

        if self.training and self._token_dropout > 0:
            eligible = padding_mask.bool()
            for special_token_id in self._special_token_ids:
                eligible = eligible & (input_ids != special_token_id)
            dropped = (
                torch.rand_like(input_ids, dtype=torch.float32) < self._token_dropout
            ) & eligible
            input_ids = input_ids.masked_fill(dropped, self._mask_token_id)

        h0: Tensor = self._embedding_dropout(self._token_embedding(input_ids))
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

        injection_scale = 1.0 / self._model_options.loop.max_iterations

        for t in range(self._model_options.loop.max_iterations):
            attn_res: Tensor = self._attn_res(h, history_states[:-1], padding_mask)
            h = h + injection_scale * attn_res
            h = self._loop(h, padding_mask)
            history_states.append(h)

            aligned = self._alignment_rotation(-(t + 1), h)
            pooled = self._mean_pooling(aligned, padding_mask)
            logits = self._classifier(
                self._classifier_dropout(self._classify_normalization(pooled))
            )

            logits_list.append(logits)

            halting_result: HaltingResult = self._halting_cell(
                t, aligned.detach(), halting_hidden_state, padding_mask
            )
            halting_hidden_state = halting_result["hidden_state"]

            hazards = halting_result["hazards"]
            hazards_list.append(hazards)
            survivals = survivals_list[-1] * (1 - hazards)
            survivals_list.append(survivals)

        timed_logits = torch.stack(logits_list)
        timed_hazards = torch.stack(hazards_list)
        timed_hazards[-1, :] = 1.0
        timed_survivals = torch.stack(survivals_list[:-1])

        return {
            "mode": "train",
            "timed_logits": timed_logits,
            "halt_probabilities": timed_hazards * timed_survivals,
            "time_rope_regularization": self._attn_res.time_rope_regularization(),
        }
