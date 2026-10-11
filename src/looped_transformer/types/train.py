from dataclasses import dataclass
from typing import Callable, Iterable

from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.data_reader import MNLIDataReader


@dataclass
class TrainingState:
    epoch: int
    global_step: int
    best_accuracy: float
    evaluation_completed: bool


@dataclass(frozen=True, slots=True)
class StageTemplate:
    # todo: separated learning rate
    name: str
    data_reader: MNLIDataReader
    loss: nn.Module
    frozen_patterns: str
    optimizer_factory: Callable[[Iterable[nn.parameter.Parameter]], Optimizer]
    scheduler_factory: Callable[[Optimizer], LRScheduler]
    epochs: int
