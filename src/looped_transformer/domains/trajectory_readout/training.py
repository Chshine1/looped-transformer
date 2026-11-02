import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import ClassVar, Literal, override

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.core.abstractions.training import (
    CollatedBatch,
    ModelTrainingCollator,
)
from looped_transformer.core.abstractions.training.data_provider import IDataProvider
from looped_transformer.core.abstractions.training.training_stage_template import (
    OptimizerFactory,
    SchedulerFactory,
    TrainingStage,
)
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.config.base import (
    ConfigPath,
    ExclusiveProbability,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.core.training.stage import StageTemplate
from looped_transformer.domains.clutrr.data_provider import CLUTRRDataOptions
from looped_transformer.domains.clutrr.types import relation_index
from looped_transformer.domains.trajectory_readout.batch import TrajectoryBatch
from looped_transformer.domains.trajectory_readout.model import (
    TrajectoryReadoutOptions,
)
from looped_transformer.utils.torch.optim import AdamWOptions, create_adam_w_optimizer
from looped_transformer.utils.torch.schedulers import SchedulerOptions, create_scheduler


class TrajectoryReadoutLossOptions(OptionsModel):
    type: Literal["cross_entropy"] = "cross_entropy"
    label_smoothing: ExclusiveProbability = 0.0


class TrajectoryReadoutStageTemplate(StageTemplate):
    type: ClassVar[str] = "trajectory_readout"
    backbone_checkpoint: ConfigPath
    data_options: CLUTRRDataOptions
    output_batch_size: PositiveInt
    model_options: TrajectoryReadoutOptions
    loss_options: TrajectoryReadoutLossOptions
    optimizer_options: AdamWOptions
    scheduler_options: SchedulerOptions


@dataclass(frozen=True, slots=True)
class TrajectoryReadoutLossInput:
    labels: Tensor


class TrajectoryReadoutCollator(
    ModelTrainingCollator[TrajectoryBatch, Tensor, Tensor, TrajectoryReadoutLossInput]
):
    @override
    def collate_batch(
        self, batch: TrajectoryBatch, device: torch.device, dtype: torch.dtype
    ) -> list[
        CollatedBatch[TrajectoryBatch, Tensor, Tensor, TrajectoryReadoutLossInput]
    ]:
        labels = torch.tensor(
            [relation_index(label) for label in batch.source.labels],
            device=device,
            dtype=torch.long,
        )
        return [
            CollatedBatch(
                batch=batch,
                task=None,
                inputs=(
                    batch.timed_hidden_states.to(device=device, dtype=dtype),
                    batch.padding_mask.to(device=device),
                ),
                targets=TrajectoryReadoutLossInput(labels=labels),
                samples_count=len(batch.source.labels),
            )
        ]


class TrajectoryReadoutCrossEntropyLoss(nn.Module):
    def __init__(self, options: TrajectoryReadoutLossOptions) -> None:
        super().__init__()
        self._label_smoothing = options.label_smoothing

    def forward(self, logits: Tensor, targets: TrajectoryReadoutLossInput) -> Tensor:
        return functional.cross_entropy(
            logits.float(),
            targets.labels,
            label_smoothing=self._label_smoothing,
        )


def create_trajectory_readout_stage(
    template: TrajectoryReadoutStageTemplate,
    data_provider: IDataProvider[TrajectoryBatch],
) -> TrainingStage[TrajectoryBatch, Tensor, TrajectoryReadoutLossInput]:
    def optimizer_factory(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        return create_adam_w_optimizer(template.optimizer_options, named_parameters)

    def scheduler_factory(optimizer: Optimizer) -> LRScheduler:
        batches = math.ceil(
            data_provider.get_samples_count() / data_provider.get_batch_size()
        )
        steps = math.ceil(batches / template.options.gradient_accumulation_steps)
        return create_scheduler(
            optimizer,
            template.scheduler_options,
            template.options.epochs * steps,
        )

    optimizer: OptimizerFactory = optimizer_factory
    scheduler: SchedulerFactory = scheduler_factory
    return TrainingStage(
        id=template.id,
        stage_options=template.options,
        data_provider=data_provider,
        loss=TypedModule(TrajectoryReadoutCrossEntropyLoss(template.loss_options)),
        optimizer_factory=optimizer,
        scheduler_factory=scheduler,
    )
