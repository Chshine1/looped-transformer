from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training.data_provider import IDataProvider
from looped_transformer.stages.config import CommonStageOptions

type OptimizerFactory = Callable[
    [Iterable[tuple[str, nn.parameter.Parameter]]], torch.optim.Optimizer
]
type SchedulerFactory = Callable[
    [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


@dataclass
class TrainingStage[TBatch, TModelOutput, TLossInput]:
    id: str
    stage_options: CommonStageOptions
    data_provider: IDataProvider[TBatch]
    loss: TypedModule[TModelOutput, TLossInput, Tensor]
    optimizer_factory: OptimizerFactory
    scheduler_factory: SchedulerFactory
