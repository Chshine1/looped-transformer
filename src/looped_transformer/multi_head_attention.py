import math

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch import Tensor


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_head = d_model // n_heads

        self._sqrt_d_head = math.sqrt(self._d_head)

        self._q_proj = nn.Linear(d_model, d_model)
        self._k_proj = nn.Linear(d_model, d_model)
        self._v_proj = nn.Linear(d_model, d_model)

        self._out_proj = nn.Linear(d_model, d_model)
        self._dropout = nn.Dropout(dropout)

    def forward(
        self, query_seq: Tensor, key_value_seq: Tensor, attention_mask: Tensor | None
    ) -> Tensor:
        batch, q_seq_length, _ = query_seq.shape
        kv_seq_length = key_value_seq.shape[1]

        q: Tensor = self._q_proj(query_seq)  # (batch, q_seq_length, feature)
        k: Tensor = self._k_proj(key_value_seq)  # (batch, kv_seq_length, feature)
        v: Tensor = self._v_proj(key_value_seq)  # (batch, kv_seq_length, feature)

        q = q.reshape(batch, q_seq_length, self._n_heads, self._d_head).transpose(1, 2)
        k = k.reshape(batch, kv_seq_length, self._n_heads, self._d_head).transpose(1, 2)
        v = v.reshape(batch, kv_seq_length, self._n_heads, self._d_head).transpose(1, 2)

        scores = q @ k.transpose(2, 3) / self._sqrt_d_head

        if attention_mask is not None:
            key_mask = attention_mask == 0
            scores = scores.masked_fill_(
                key_mask[:, None, None, :], torch.finfo(scores.dtype).min
            )

        attention = functional.softmax(scores, dim=3)
        attention = self._dropout(attention)

        out = attention @ v
        out = out.transpose(1, 2).contiguous().view(batch, q_seq_length, self._d_model)
        return self._out_proj(out)
