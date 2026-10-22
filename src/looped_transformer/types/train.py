from abc import ABC
from abc import abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.data_reader import MNLIDataProvider


@dataclass
class TrainingState:
    stage_id: str
    epoch: int
    global_step: int
    best_accuracy: float


@dataclass(frozen=True, slots=True)
class StageTemplate:
    id: str
    data_provider: MNLIDataProvider
    loss: nn.Module
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


class IDataProvider[TBatch](ABC):
    @abstractmethod
    def get_batches_count(self) -> int:
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def reset_source(self, shuffle: bool, seed: int | None) -> None:
        pass

    @abstractmethod
    def fetch(self) -> TBatch | None:
        pass
