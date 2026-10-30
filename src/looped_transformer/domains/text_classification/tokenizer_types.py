from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class TokenizerInfo:
    vocab_size: int
    pad_token_id: int
    mask_token_id: int = 103
    special_token_ids: tuple[int, ...] = (0, 100, 101, 102, 103)


@dataclass
class TokenizeResult:
    input_ids: Tensor
    attention_mask: Tensor
    token_type_ids: Tensor
