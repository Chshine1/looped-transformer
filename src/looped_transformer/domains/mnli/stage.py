"""Training-stage registration for the looped MNLI model."""

import math
import re
from collections.abc import Iterable
from typing import ClassVar

import torch
from pydantic import Field
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.core.abstractions.model import ModelForwardOutput, TypedModule
from looped_transformer.core.abstractions.training.data_provider import IDataProvider
from looped_transformer.core.abstractions.training.training_stage_template import (
    OptimizerFactory,
    SchedulerFactory,
)
from looped_transformer.core.torch.optim import AdamWOptions
from looped_transformer.core.torch.schedulers import (
    SchedulerOptions,
    WarmupCosineOptions,
)
from looped_transformer.core.training.stage import StageTemplate
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.mnli.batch import MNLIBatch
from looped_transformer.domains.mnli.data import MNLIDataOptions, MNLIDataProvider
from looped_transformer.domains.mnli.loss import (
    AnchoredSampledLoopCrossEntropyLoss,
    AnchoredSampledLoopLossOptions,
)


class LoopedMNLIStageTemplate(StageTemplate):
    type: ClassVar[str] = "looped_mnli"
    data_options: MNLIDataOptions = Field(validation_alias="data")
    loss_options: AnchoredSampledLoopLossOptions = Field(validation_alias="loss")
    optimizer_options: AdamWOptions = Field(validation_alias="optimizer")
    scheduler_options: SchedulerOptions = Field(validation_alias="scheduler")


def _get_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    options: AdamWOptions,
) -> list[dict]:
    patterns = [
        (re.compile(module_pattern), learning_rate)
        for module_pattern, learning_rate in options.learning_rates.items()
    ]
    groups: dict[tuple[float, float], list[nn.parameter.Parameter]] = {}
    unmatched_modules: set[str] = set()

    for parameter_name, parameter in named_parameters:
        module_name, _, _ = parameter_name.rpartition(".")
        learning_rate = next(
            (
                learning_rate
                for pattern, learning_rate in patterns
                if pattern.fullmatch(module_name)
            ),
            None,
        )
        if learning_rate is None:
            unmatched_modules.add(module_name or "<root>")
            continue
        weight_decay = options.weight_decay if parameter.ndim >= 2 else 0.0
        groups.setdefault((learning_rate, weight_decay), []).append(parameter)

    if unmatched_modules:
        names = ", ".join(sorted(unmatched_modules))
        raise ValueError(f"No learning rate configured for modules: {names}")
    if not groups:
        raise ValueError("The stage has no trainable parameters")

    return [
        {
            "params": parameters,
            "lr": learning_rate,
            "weight_decay": weight_decay,
        }
        for (learning_rate, weight_decay), parameters in groups.items()
    ]


def _get_data_provider(template: LoopedMNLIStageTemplate) -> IDataProvider[MNLIBatch]:
    return MNLIDataProvider(template.data_options)


def _get_loss(
    template: LoopedMNLIStageTemplate,
) -> TypedModule[ModelForwardOutput, Tensor, Tensor]:
    loss = AnchoredSampledLoopCrossEntropyLoss(template.loss_options)
    return TypedModule(loss)


def _get_optimizer_factory(template: LoopedMNLIStageTemplate) -> OptimizerFactory:
    def get_optimizer(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        optimizer = template.optimizer_options
        return torch.optim.AdamW(
            _get_parameter_groups(named_parameters, optimizer),
            betas=(optimizer.beta1, optimizer.beta2),
            eps=optimizer.epsilon,
            weight_decay=optimizer.weight_decay,
        )

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
        scheduler = template.scheduler_options
        warmup_steps = max(1, int(scheduler.warmup_ratio * total_steps))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=scheduler.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        if isinstance(scheduler, WarmupCosineOptions):
            cosine_steps = max(1, total_steps - warmup_steps)
            minimum_factor = scheduler.minimum_learning_rate_factor

            def cosine_factor(step: int) -> float:
                progress = min(step, cosine_steps) / cosine_steps
                return (
                    minimum_factor
                    + (1 - minimum_factor) * (1 + math.cos(math.pi * progress)) / 2
                )

            after_warmup = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=cosine_factor
            )
        else:
            after_warmup = torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=1.0,
                total_iters=max(1, total_steps - warmup_steps),
            )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, after_warmup], [warmup_steps]
        )

    return get_scheduler


def register_looped_mnli_stage_template(
    stage_registry: TrainingStageRegistry[MNLIBatch, ModelForwardOutput, Tensor],
) -> None:
    stage_registry.register(
        LoopedMNLIStageTemplate,
        _get_data_provider,
        _get_loss,
        _get_optimizer_factory,
        _get_scheduler_factory,
    )
