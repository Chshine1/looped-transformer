from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TokenizerOptions:
    pretrained_model_name: str
    vocab_size: int | None = None
    pad_token_id: int | None = None
