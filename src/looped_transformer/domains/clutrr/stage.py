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
from looped_transformer.core.training.stage import StageTemplate
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.clutrr.data_provider import (
    CLUTRRDataOptions,
    CLUTRRDataProvider,
)
from looped_transformer.domains.clutrr.types import CLUTRRBatch
from looped_transformer.domains.text_classification.loss import (
    AnchoredSampledLoopCrossEntropyLoss,
    AnchoredSampledLoopLossOptions,
    BackboneLossInput,
)
from looped_transformer.domains.text_classification.types import ModelForwardOutput
from looped_transformer.utils.torch.optim import AdamWOptions, create_adam_w_optimizer
from looped_transformer.utils.torch.schedulers import SchedulerOptions, create_scheduler


class LoopedCLUTRRStageTemplate(StageTemplate):
    type: ClassVar[str] = "looped_clutrr"
    data_options: CLUTRRDataOptions
    loss_options: AnchoredSampledLoopLossOptions
    optimizer_options: AdamWOptions
    scheduler_options: SchedulerOptions


def _get_data_provider(
    template: LoopedCLUTRRStageTemplate,
) -> IDataProvider[CLUTRRBatch]:
    return CLUTRRDataProvider(template.data_options)


def _get_loss(
    template: LoopedCLUTRRStageTemplate,
) -> TypedModule[ModelForwardOutput, BackboneLossInput, Tensor]:
    if template.loss_options.distillation_scale != 0:
        raise ValueError("External NLI distillation is not defined for CLUTRR")
    return TypedModule(AnchoredSampledLoopCrossEntropyLoss(template.loss_options))


def _get_optimizer_factory(template: LoopedCLUTRRStageTemplate) -> OptimizerFactory:
    def get_optimizer(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        return create_adam_w_optimizer(template.optimizer_options, named_parameters)

    return get_optimizer


def _get_scheduler_factory(
    template: LoopedCLUTRRStageTemplate, data_provider: IDataProvider[CLUTRRBatch]
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


def register_looped_clutrr_stage_template(
    registry: TrainingStageRegistry[CLUTRRBatch, ModelForwardOutput, BackboneLossInput],
) -> None:
    registry.register(
        LoopedCLUTRRStageTemplate,
        _get_data_provider,
        _get_loss,
        _get_optimizer_factory,
        _get_scheduler_factory,
    )
