from typing import Protocol, Self

from torch import device as _torch_device
from torch import dtype as _torch_dtype
from torch import nn


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
        device: _torch_device | int | str | None = None,
        dtype: _torch_dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        self._torch.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self
