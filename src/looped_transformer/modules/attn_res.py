import math

import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.modules.residual_scale import ResidualScale
from looped_transformer.utils.shared_cache import PositionalRoPECache


class AttnRes(nn.Module):
    _rope_cache: PositionalRoPECache

    def __init__(
        self,
        positional_rope_cache: PositionalRoPECache,
        d_model: int,
        n_heads: int,
        max_iterations: int,
        dropout: float,
    ):
        super().__init__()
        object.__setattr__(self, "_rope_cache", positional_rope_cache)

        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self._d_model = d_model
        self._n_heads = n_heads
        self._d_head = d_model // n_heads
        if self._d_head % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE")

        self._sqrt_d_head = math.sqrt(self._d_head)

        self._q_proj = nn.Linear(d_model, d_model)
        self._k_proj = nn.Linear(d_model, d_model)
        self._v_proj = nn.Linear(d_model, d_model)

        self._history_bias = nn.Parameter(torch.zeros(max_iterations))

        self._out_proj = nn.Linear(d_model, d_model)
        self._dropout = nn.Dropout(dropout)
        self._residual_scale = ResidualScale()

    def time_rope_regularization(self) -> Tensor:
        penalties = []
        for projection in (self._q_proj, self._k_proj):
            weight = projection.weight.float()
            blocks = weight.reshape(
                self._n_heads, self._d_head, self._n_heads, self._d_head
            ).permute(0, 2, 1, 3)
            rr, ri = blocks.chunk(2, dim=-2)[0].chunk(2, dim=-1)
            ir, ii = blocks.chunk(2, dim=-2)[1].chunk(2, dim=-1)
            a, b = (rr + ii) / 2, (ir - ri) / 2
            complex_linear = torch.cat(
                [torch.cat([a, -b], dim=-1), torch.cat([b, a], dim=-1)],
                dim=-2,
            )
            head_mask = torch.eye(self._n_heads, device=weight.device)
            target = complex_linear * head_mask[:, :, None, None]
            penalty = (blocks - target).square().sum() / self._d_model
            if projection.bias is not None:
                penalty = penalty + projection.bias.float().square().mean()
            penalties.append(penalty)
        return torch.stack(penalties).mean()

    def _apply_rope(self, x: Tensor, seq_length: int) -> Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        complex_x = torch.complex(x1, x2) * self._rope_cache.rotations[:seq_length]
        return torch.cat([complex_x.real, complex_x.imag], dim=-1)

    def forward(
        self,
        query_seq: Tensor,
        history: list[Tensor] | tuple[Tensor],
        attention_mask: Tensor,
    ) -> Tensor:
        history_length = len(history)
        if history_length == 0:
            return torch.zeros_like(query_seq)

        batch, q_seq_length, _ = query_seq.shape
        history_tensor = torch.stack(history, dim=0)
        history_seq_length = history_tensor.size(2)

        q: Tensor = self._q_proj(query_seq)
        k: Tensor = self._k_proj(history_tensor)
        v: Tensor = self._v_proj(history_tensor)

        q = q.reshape(batch, q_seq_length, self._n_heads, self._d_head).transpose(1, 2)
        k = k.reshape(
            history_length, batch, history_seq_length, self._n_heads, self._d_head
        ).transpose(2, 3)
        v = v.reshape(
            history_length, batch, history_seq_length, self._n_heads, self._d_head
        ).transpose(2, 3)

        q = self._apply_rope(q, q_seq_length)
        k = self._apply_rope(k, history_seq_length)

        scores = q @ k.transpose(-1, -2) / self._sqrt_d_head
        scores = scores.masked_fill_(
            (attention_mask == 0)[None, :, None, None, :], torch.finfo(scores.dtype).min
        )

        history_attention = functional.softmax(scores, dim=-1)
        history_attention = self._dropout(history_attention)
        history_out = history_attention @ v

        history_logits = self._history_bias[:history_length]
        history_weights = torch.softmax(history_logits, dim=0)

        out = (history_out * history_weights.reshape(-1, 1, 1, 1, 1)).sum(dim=0)
        out = out.transpose(1, 2).reshape(batch, q_seq_length, self._d_model)
        return self._residual_scale() * self._out_proj(out)
