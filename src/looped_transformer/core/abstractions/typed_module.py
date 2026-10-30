from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Protocol, Self

from torch import device as _torch_device
from torch import dtype as _torch_dtype
from torch import nn

# noinspection protected-member
from torch.nn.modules.module import _IncompatibleKeys

if TYPE_CHECKING:
    from looped_transformer.core.modules.branched_module import BranchedModule


class IModule[*TIn, TOut](Protocol):
    def forward(self, *args: *TIn) -> TOut: ...
    def __call__(self, *args: *TIn) -> TOut: ...


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
        module: BranchedModule[*TIn, TOut],
    ) -> None:
        self._module = module

    @property
    def torch(self) -> nn.Module:
        return self._module

    def __call__(self, *args: *TIn, task: str | None) -> TOut:
        return self._module(*args, task)

    def state_dict(self) -> dict[str, Any]:
        return self._module.state_dict()

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        *,
        strict_main: bool = True,
        strict_branches: bool = False,
        assign: bool = False,
    ) -> _IncompatibleKeys:
        result = self._module.load_state_dict(
            state_dict,
            strict=False,
            assign=assign,
        )

        def is_branch_key(key: str) -> bool:
            return ".branches." in key

        main_missing = [k for k in result.missing_keys if not is_branch_key(k)]
        main_unexpected = [k for k in result.unexpected_keys if not is_branch_key(k)]
        branch_missing = [k for k in result.missing_keys if is_branch_key(k)]
        branch_unexpected = [k for k in result.unexpected_keys if is_branch_key(k)]

        errors: list[str] = []
        if strict_main:
            if main_missing:
                errors.append(f"missing main keys: {main_missing}")
            if main_unexpected:
                errors.append(f"unexpected main keys: {main_unexpected}")
        if strict_branches:
            if branch_missing:
                errors.append(f"missing branch keys: {branch_missing}")
            if branch_unexpected:
                errors.append(f"unexpected branch keys: {branch_unexpected}")
        if errors:
            raise RuntimeError(
                f"Error(s) in loading state_dict for "
                f"{type(self._module).__name__}:\n" + "\n".join(errors)
            )

        return result

    def to(
        self,
        device: _torch_device | int | str | None = None,
        dtype: _torch_dtype | None = None,
        non_blocking: bool = False,
    ) -> Self:
        self._module.to(device=device, dtype=dtype, non_blocking=non_blocking)
        return self
