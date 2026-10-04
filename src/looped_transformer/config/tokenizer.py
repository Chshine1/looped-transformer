from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerOptions:
    vocab_size: int
    pad_token_id: int
