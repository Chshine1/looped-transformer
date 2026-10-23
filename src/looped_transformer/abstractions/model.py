from typing import Literal
from typing import NotRequired
from typing import Protocol
from typing import TypedDict

from torch import Tensor, nn


class IModule[*TIn, TOut](Protocol):
    def forward(self, *args: *TIn) -> TOut: ...
    def __call__(self, *args: *TIn) -> TOut: ...


class TypedModule[*TIn, TOut]:
    def __init__(self, model: IModule[*TIn, TOut]) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"Expected an nn.Module, got {type(model).__name__}")
        self._torch: nn.Module = model
        self._typed: IModule[*TIn, TOut] = model

    @property
    def torch(self) -> nn.Module:
        return self._torch

    def __call__(self, *args: *TIn) -> TOut:
        return self._typed(*args)

    def to(
        self,
        device: torch.device | int | str | None = None,
        dtype: torch.dtype | None = None,
        non_blocking: bool = False,
    ) -> TypedModule[*TIn, TOut]:
        self._torch.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self


class ModelTrainOutput(TypedDict):
    mode: Literal["train"]
    timed_logits: Tensor
    halt_probabilities: Tensor
    time_rope_regularization: NotRequired[Tensor]


ModelForwardOutput = ModelTrainOutput
