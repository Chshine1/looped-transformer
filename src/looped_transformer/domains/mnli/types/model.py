from typing import Literal, NotRequired, TypedDict

from torch import Tensor


class ModelForwardOutput(TypedDict):
    mode: Literal["train"]
    timed_logits: Tensor
    halt_probabilities: Tensor
    time_rope_regularization: NotRequired[Tensor]
