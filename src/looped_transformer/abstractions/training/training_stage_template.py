from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch import Tensor
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training.data_provider import IDataProvider


@dataclass(frozen=True, slots=True)
class StageTemplate[TBatch, TModelOutput, TLossInput]:
    id: str
    data_provider: IDataProvider[TBatch]
    loss: TypedModule[TModelOutput, TLossInput, Tensor]
    frozen_patterns: str
    evaluation_patterns: str
    optimizer_factory: Callable[
        [Iterable[tuple[str, nn.parameter.Parameter]]], Optimizer
    ]
    scheduler_factory: Callable[[Optimizer], LRScheduler]
    epochs: int
    max_gradient_norm: float
    gradient_accumulation_steps: int
    save_every_epochs: int
