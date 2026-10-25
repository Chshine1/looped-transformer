from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from looped_transformer.core.abstractions.training.data_provider import IDataProvider
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.training.stage import StageOptions

type OptimizerFactory = Callable[
    [Iterable[tuple[str, nn.parameter.Parameter]]], torch.optim.Optimizer
]
type SchedulerFactory = Callable[
    [torch.optim.Optimizer], torch.optim.lr_scheduler.LRScheduler
]


@dataclass
class TrainingStage[TBatch, TModelOutput, TLossInput]:
    id: str
    stage_options: StageOptions
    data_provider: IDataProvider[TBatch]
    loss: TypedModule[TModelOutput, TLossInput, Tensor]
    optimizer_factory: OptimizerFactory
    scheduler_factory: SchedulerFactory
