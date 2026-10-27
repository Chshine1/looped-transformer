import math
from collections.abc import Iterable
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
from looped_transformer.core.config.base import ConfigPath
from looped_transformer.core.training.stage import StageTemplate
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.guided_backbone.loss import (
    ControllerWeightedLoopCrossEntropyLoss,
    ControllerWeightedLoopLossOptions,
)
from looped_transformer.domains.guided_backbone.model import (
    GuidanceWeightOptions,
    GuidedBackboneOutput,
)
from looped_transformer.domains.mnli.services.data_provider import (
    MNLIDataOptions,
    MNLIDataProvider,
)
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.utils.torch.optim import AdamWOptions, create_adam_w_optimizer
from looped_transformer.utils.torch.schedulers import SchedulerOptions, create_scheduler


class ControllerGuidedBackboneStageTemplate(StageTemplate):
    type: ClassVar[str] = "controller_guided_backbone"
    source_checkpoint: ConfigPath
    controller_checkpoint: ConfigPath
    data_options: MNLIDataOptions
    guidance_options: GuidanceWeightOptions
    loss_options: ControllerWeightedLoopLossOptions
    optimizer_options: AdamWOptions
    scheduler_options: SchedulerOptions


def _get_data_provider(
    template: ControllerGuidedBackboneStageTemplate,
) -> IDataProvider[MNLIBatch]:
    return MNLIDataProvider(template.data_options)


def _get_loss(
    template: ControllerGuidedBackboneStageTemplate,
) -> TypedModule[GuidedBackboneOutput, Tensor, Tensor]:
    return TypedModule(ControllerWeightedLoopCrossEntropyLoss(template.loss_options))


def _get_optimizer_factory(
    template: ControllerGuidedBackboneStageTemplate,
) -> OptimizerFactory:
    def get_optimizer(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        return create_adam_w_optimizer(template.optimizer_options, named_parameters)

    return get_optimizer


def _get_scheduler_factory(
    template: ControllerGuidedBackboneStageTemplate,
    data_provider: IDataProvider[MNLIBatch],
) -> SchedulerFactory:
    def get_scheduler(optimizer: Optimizer) -> LRScheduler:
        batches = math.ceil(
            data_provider.get_samples_count() / data_provider.get_batch_size()
        )
        steps = math.ceil(batches / template.options.gradient_accumulation_steps)
        return create_scheduler(
            optimizer, template.scheduler_options, template.options.epochs * steps
        )

    return get_scheduler


def register_controller_guided_backbone_stage(
    registry: TrainingStageRegistry[MNLIBatch, GuidedBackboneOutput, Tensor],
) -> None:
    registry.register(
        ControllerGuidedBackboneStageTemplate,
        _get_data_provider,
        _get_loss,
        _get_optimizer_factory,
        _get_scheduler_factory,
    )
