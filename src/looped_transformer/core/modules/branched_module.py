from abc import abstractmethod
from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, final, override

from torch import nn

from looped_transformer.core.abstractions.typed_module import IModule, TypedModule


class _BranchesRouter[*TIn, TOut](nn.Module, IModule[*TIn, TOut]):
    def __init__(
        self,
        main: nn.Module,
        branches: Mapping[str, nn.Module],
        task_context_var: ContextVar[str | None],
    ) -> None:
        super().__init__()
        self.main = main
        self.branches = nn.ModuleDict(dict(branches))
        self._task_context_var = task_context_var

    @override
    def forward(self, *args: *TIn) -> TOut:
        task = self._task_context_var.get()
        if task is None:
            return self.main(*args)
        try:
            branch = self.branches[task]
        except KeyError:
            raise KeyError(
                f"no branch registered for task {task!r}; "
                f"available: {list(self.branches)!r}"
            ) from None
        return branch(*args)

    def extra_repr(self) -> str:
        return f"branches={list(self.branches)!r}"


class BranchedModule[*TIn, TOut](nn.Module, IModule[*TIn, str | None, TOut]):
    def __init__(self) -> None:
        super().__init__()
        self._task_context_var: ContextVar[str | None] = ContextVar(
            f"branched_task_{id(self)}", default=None
        )

    def _create_branched[*TBranchIn, TBranchOut](
        self,
        main_branch: TypedModule[*TBranchIn, TBranchOut],
        side_branches: Mapping[str, TypedModule[*TBranchIn, TBranchOut]],
    ) -> IModule[*TBranchIn, TBranchOut]:
        return _BranchesRouter(
            main=main_branch.torch,
            branches={task: branch.torch for task, branch in side_branches.items()},
            task_context_var=self._task_context_var,
        )

    @final
    @override
    def forward(self, *args: *tuple[*TIn, str | None]) -> TOut:
        if not args:
            raise TypeError(
                f"{type(self).__name__}.forward() requires at least the task argument"
            )
        task = args[-1]
        if task is not None and not isinstance(task, str):
            raise TypeError(f"task must be str | None, got {type(task).__name__}")
        token = self._task_context_var.set(task)
        try:
            return self._forward_task(*args[:-1])
        finally:
            self._task_context_var.reset(token)

    @abstractmethod
    def _forward_task(self, *args: *TIn) -> TOut: ...

    def main_state_dict(self, name: str) -> dict[str, Any]:
        router = self._modules.get(name)
        if not isinstance(router, _BranchesRouter):
            raise KeyError(f"no branched router registered under {name!r}")
        prefix = f"{name}.main."
        return {
            key[len(prefix) :]: value
            for key, value in self.state_dict().items()
            if key.startswith(prefix)
        }
