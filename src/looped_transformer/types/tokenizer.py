from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True, slots=True)
class TokenizerInfo:
    vocab_size: int
    pad_token_id: int


@dataclass
class TokenizeResult:
    input_ids: Tensor
    attention_mask: Tensor
