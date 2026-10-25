from typing import TypedDict

from torch import Tensor


class HaltingResult(TypedDict):
    hidden_state: Tensor
    hazards: Tensor
