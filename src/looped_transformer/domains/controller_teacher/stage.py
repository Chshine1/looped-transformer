import math
from collections.abc import Callable, Iterable
from typing import ClassVar

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.core.abstractions.training import IDataProvider
from looped_transformer.core.abstractions.training.training_stage_template import (
    OptimizerFactory,
    SchedulerFactory,
)
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.config.base import (
    ConfigPath,
    NonNegativeFloat,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.core.training.stage import StageTemplate
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.controller_teacher.batch import (
    ControllerTeacherBatch,
)
from looped_transformer.domains.controller_teacher.controller_teacher_training import (
    ControllerTeacherLoss,
    ControllerTeacherLossInput,
)
from looped_transformer.domains.mnli.services.data_provider import MNLIDataOptions
from looped_transformer.utils.torch.optim import AdamWOptions, create_adam_w_optimizer
from looped_transformer.utils.torch.schedulers import SchedulerOptions, create_scheduler


class ControllerTeacherLossOptions(OptionsModel):
    incorrect_prediction_cost: NonNegativeFloat = 0.5


class ControllerTeacherStageTemplate(StageTemplate):
    type: ClassVar[str] = "controller_teacher"
    source_checkpoint: ConfigPath
    data_options: MNLIDataOptions
    training_batch_size: PositiveInt
    loss_options: ControllerTeacherLossOptions
    optimizer_options: AdamWOptions
    scheduler_options: SchedulerOptions


def _get_loss(
    template: ControllerTeacherStageTemplate,
) -> TypedModule[Tensor, ControllerTeacherLossInput, Tensor]:
    return TypedModule(
        ControllerTeacherLoss(
            template.loss_options.incorrect_prediction_cost,
        )
    )


def _get_optimizer_factory(
    template: ControllerTeacherStageTemplate,
) -> OptimizerFactory:
    def get_optimizer(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        return create_adam_w_optimizer(template.optimizer_options, named_parameters)

    return get_optimizer


def _get_scheduler_factory(
    template: ControllerTeacherStageTemplate,
    data_provider: IDataProvider[ControllerTeacherBatch],
) -> SchedulerFactory:
    def get_scheduler(optimizer: Optimizer) -> LRScheduler:
        batches_per_epoch = math.ceil(
            data_provider.get_samples_count() / data_provider.get_batch_size()
        )
        steps_per_epoch = math.ceil(
            batches_per_epoch / template.options.gradient_accumulation_steps
        )
        total_steps = template.options.epochs * steps_per_epoch
        return create_scheduler(optimizer, template.scheduler_options, total_steps)

    return get_scheduler


def register_controller_teacher_stage_template(
    stage_registry: TrainingStageRegistry[
        ControllerTeacherBatch,
        Tensor,
        ControllerTeacherLossInput,
    ],
    data_provider_factory: Callable[
        [ControllerTeacherStageTemplate], IDataProvider[ControllerTeacherBatch]
    ],
) -> None:
    stage_registry.register(
        ControllerTeacherStageTemplate,
        data_provider_factory,
        _get_loss,
        _get_optimizer_factory,
        _get_scheduler_factory,
    )
