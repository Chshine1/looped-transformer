import math
from collections.abc import Iterable
from typing import ClassVar

from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.core.abstractions.training.data_provider import IDataProvider
from looped_transformer.core.abstractions.training.training_stage_template import (
    OptimizerFactory,
    SchedulerFactory,
)
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.training.stage import StageTemplate
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.mnli.modules.anchored_sampled_ce_loss import (
    AnchoredSampledLoopCrossEntropyLoss,
    AnchoredSampledLoopLossOptions,
    BackboneLossInput,
)
from looped_transformer.domains.mnli.services.data_provider import (
    MNLIDataOptions,
    MNLIDataProvider,
)
from looped_transformer.domains.mnli.training.backbone_collator import (
    ExternalTeacherOptions,
)
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.mnli.types.model import ModelForwardOutput
from looped_transformer.utils.torch.optim import AdamWOptions, create_adam_w_optimizer
from looped_transformer.utils.torch.schedulers import SchedulerOptions, create_scheduler


class LoopedMNLIStageTemplate(StageTemplate):
    type: ClassVar[str] = "looped_mnli"
    data_options: MNLIDataOptions
    teacher: ExternalTeacherOptions
    loss_options: AnchoredSampledLoopLossOptions
    optimizer_options: AdamWOptions
    scheduler_options: SchedulerOptions


def _get_data_provider(template: LoopedMNLIStageTemplate) -> IDataProvider[MNLIBatch]:
    return MNLIDataProvider(template.data_options)


def _get_loss(
    template: LoopedMNLIStageTemplate,
) -> TypedModule[ModelForwardOutput, BackboneLossInput, Tensor]:
    loss = AnchoredSampledLoopCrossEntropyLoss(template.loss_options)
    return TypedModule(loss)


def _get_optimizer_factory(template: LoopedMNLIStageTemplate) -> OptimizerFactory:
    def get_optimizer(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        return create_adam_w_optimizer(template.optimizer_options, named_parameters)

    return get_optimizer


def _get_scheduler_factory(
    template: LoopedMNLIStageTemplate, data_provider: IDataProvider[MNLIBatch]
) -> SchedulerFactory:
    def get_scheduler(optimizer: Optimizer) -> LRScheduler:
        row_count = data_provider.get_samples_count()
        batches_per_epoch = math.ceil(row_count / template.data_options.batch_size)
        steps_per_epoch = math.ceil(
            batches_per_epoch / template.options.gradient_accumulation_steps
        )
        total_steps = template.options.epochs * steps_per_epoch
        return create_scheduler(optimizer, template.scheduler_options, total_steps)

    return get_scheduler


def register_looped_mnli_stage_template(
    stage_registry: TrainingStageRegistry[
        MNLIBatch, ModelForwardOutput, BackboneLossInput
    ],
) -> None:
    stage_registry.register(
        LoopedMNLIStageTemplate,
        _get_data_provider,
        _get_loss,
        _get_optimizer_factory,
        _get_scheduler_factory,
    )
