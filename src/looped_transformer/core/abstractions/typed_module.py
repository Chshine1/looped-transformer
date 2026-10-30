from typing import Protocol, Self

from torch import device as _torch_device
from torch import dtype as _torch_dtype
from torch import nn


class IModule[*TIn, TOut](Protocol):
    def forward(self, *args: *TIn) -> TOut: ...
    def __call__(self, *args: *TIn) -> TOut: ...


class TypedModel[*TIn, TOut](Protocol):
    @property
    def torch(self) -> nn.Module: ...

    def forward_task(self, *args: *TIn, task: str | None) -> TOut: ...


class TypedModule[*TIn, TOut]:
    def __init__(self, module: IModule[*TIn, TOut]) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected an nn.Module, got {type(module).__name__}")
        self._torch: nn.Module = module

    @property
    def torch(self) -> nn.Module:
        return self._torch

    def __call__(self, *args: *TIn) -> TOut:
        return self._torch(*args)

    def forward_task(self, *args: *TIn, task: str | None) -> TOut:
        if task is not None:
            raise ValueError(
                f"task {task!r} requires a TypedBranchedModule, got "
                f"{type(self._torch).__name__}"
            )
        return self(*args)

    def to(
        self,
        device: _torch_device | int | str | None = None,
        dtype: _torch_dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        self._torch.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self


class TypedBranchedModule[*TIn, TOut]:
    def __init__(
        self,
        module: IModule[*tuple[*TIn, str | None], TOut],
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError(f"Expected an nn.Module, got {type(module).__name__}")
        self._module = module
        self._torch: nn.Module = module

    @property
    def torch(self) -> nn.Module:
        return self._torch

    def __call__(self, *args: *TIn, task: str | None) -> TOut:
        return self._module(*args, task)

    def forward_task(self, *args: *TIn, task: str | None) -> TOut:
        return self(*args, task=task)

    def to(
        self,
        device: _torch_device | int | str | None = None,
        dtype: _torch_dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        self._torch.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self
