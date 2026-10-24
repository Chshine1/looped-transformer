import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor

from looped_transformer.config.base import (
    ExclusiveProbability,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.utils.positional_rope_cache import PositionalRoPECache
from looped_transformer.utils.rotary import apply_rotary


class AttentionOptions(OptionsModel):
    model_dimension: PositiveInt
    heads_count: PositiveInt
    dropout: ExclusiveProbability


class RoPEMultiHeadAttention(nn.Module):
    _rope_cache: PositionalRoPECache

    def __init__(
        self,
        positional_rope_cache: PositionalRoPECache,
        options: AttentionOptions,
    ):
        super().__init__()
        object.__setattr__(self, "_rope_cache", positional_rope_cache)

        d_model = options.model_dimension
        assert d_model % options.heads_count == 0, (
            "d_model must be divisible by n_heads"
        )
        self._d_model = d_model
        self._n_heads = options.heads_count
        self._d_head = d_model // options.heads_count
        if self._d_head % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE")

        self._sqrt_d_head = math.sqrt(self._d_head)

        self._q_proj = nn.Linear(d_model, d_model)
        self._k_proj = nn.Linear(d_model, d_model)
        self._v_proj = nn.Linear(d_model, d_model)

        self._out_proj = nn.Linear(d_model, d_model)
        self._dropout = nn.Dropout(options.dropout)

    def _apply_rope(self, x: Tensor) -> Tensor:
        seq_length = x.size(2)
        return apply_rotary(
            x, self._rope_cache.cosine[:seq_length], self._rope_cache.sine[:seq_length]
        )

    def forward(
        self, query_seq: Tensor, key_value_seq: Tensor, attention_mask: Tensor
    ) -> Tensor:
        batch, q_seq_length, _ = query_seq.shape
        kv_seq_length = key_value_seq.shape[1]

        q: Tensor = self._q_proj(query_seq)
        k: Tensor = self._k_proj(key_value_seq)
        v: Tensor = self._v_proj(key_value_seq)

        q = q.reshape(batch, q_seq_length, self._n_heads, self._d_head).transpose(1, 2)
        k = k.reshape(batch, kv_seq_length, self._n_heads, self._d_head).transpose(1, 2)
        v = v.reshape(batch, kv_seq_length, self._n_heads, self._d_head).transpose(1, 2)

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        scores = q @ k.transpose(2, 3) / self._sqrt_d_head
        scores = scores.masked_fill_(
            (attention_mask == 0)[:, None, None, :], torch.finfo(scores.dtype).min
        )

        attention = functional.softmax(scores, dim=-1)
        attention = self._dropout(attention)

        out: Tensor = attention @ v
        out = out.transpose(1, 2).reshape(batch, q_seq_length, self._d_model)
        return self._out_proj(out)
