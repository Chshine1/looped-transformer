"""Shared output types for adaptive halting modules."""

from typing import TypedDict

from torch import Tensor


class HaltingResult(TypedDict):
    hidden_state: Tensor
    hazards: Tensor
