import math
import re
from collections.abc import Iterable
from typing import Annotated, ClassVar, Literal

import torch
from pydantic import BeforeValidator, Field, model_validator
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.config.base import (
    ExclusiveProbability,
    NonNegativeFloat,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.config.torch.optimizer import AdamWOptions
from looped_transformer.config.torch.scheduler import SchedulerOptions
from looped_transformer.config.torch.scheduler import WarmupCosineOptions
from looped_transformer.config.training import CommonStageOptions, StageOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.loss import (
    AnchoredSampledLoopCrossEntropyLoss,
)
from looped_transformer.stages.registry import register_stage
from looped_transformer.types.train import StageTemplate


class AnchoredSampledLoopLossOptions(OptionsModel):
    type: Literal["anchored_sampled_loop_cross_entropy"]
    sampled_loops: Annotated[
        tuple[PositiveInt, ...], BeforeValidator(tuple), Field(min_length=1)
    ]
    sampled_weights: Annotated[
        tuple[NonNegativeFloat, ...], BeforeValidator(tuple), Field(min_length=1)
    ]
    sampled_loss_scale: NonNegativeFloat
    label_smoothing: ExclusiveProbability = 0.0
    time_rope_regularization_scale: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def validate_samples(self) -> "AnchoredSampledLoopLossOptions":
        if len(self.sampled_loops) != len(self.sampled_weights):
            raise ValueError("sampled_loops and sampled_weights must have equal length")
        if len(set(self.sampled_loops)) != len(self.sampled_loops):
            raise ValueError("sampled_loops must not contain duplicates")
        if not any(self.sampled_weights):
            raise ValueError("at least one sampled-loop weight must be positive")
        return self


LossOptions = AnchoredSampledLoopLossOptions


class LoopedMNLIStageOptions(StageOptions):
    stage_type: ClassVar[str] = "looped_mnli"

    common: CommonStageOptions
    optimizer: AdamWOptions
    scheduler: SchedulerOptions
    loss: LossOptions
    type: Literal["looped_mnli"]


def _build_stage_template(
    template_id: str,
    options: StageOptions,
) -> StageTemplate:
    if not isinstance(options, LoopedMNLIStageOptions):
        raise TypeError("looped_mnli requires LoopedMNLIStageOptions")

    data_options = options.common.data
    data_reader = MNLIDataReader(data_options)

    def get_optimizer(
        named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    ) -> Optimizer:
        optimizer = options.optimizer
        return torch.optim.AdamW(
            _get_parameter_groups(named_parameters, optimizer),
            betas=(optimizer.beta1, optimizer.beta2),
            eps=optimizer.epsilon,
            weight_decay=optimizer.weight_decay,
        )

    def get_scheduler(optimizer: Optimizer) -> LRScheduler:
        row_count = data_reader.count_examples()
        batches_per_epoch = math.ceil(row_count / data_options.batch_size)
        steps_per_epoch = math.ceil(
            batches_per_epoch / options.common.gradient_accumulation_steps
        )
        total_steps = options.common.epochs * steps_per_epoch
        scheduler = options.scheduler
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

    common = options.common
    return StageTemplate(
        id=template_id,
        data_reader=data_reader,
        loss=_build_loss(options.loss),
        frozen_patterns=common.frozen_patterns,
        evaluation_patterns=common.evaluation_patterns,
        optimizer_factory=get_optimizer,
        scheduler_factory=get_scheduler,
        epochs=common.epochs,
        max_gradient_norm=common.max_gradient_norm,
        gradient_accumulation_steps=common.gradient_accumulation_steps,
        save_every_epochs=common.save_every_epochs,
    )


def _build_loss(options: LossOptions) -> nn.Module:
    if isinstance(options, AnchoredSampledLoopLossOptions):
        return AnchoredSampledLoopCrossEntropyLoss(
            sampled_loops=options.sampled_loops,
            sampled_weights=options.sampled_weights,
            sampled_loss_scale=options.sampled_loss_scale,
            label_smoothing=options.label_smoothing,
            time_rope_regularization_scale=options.time_rope_regularization_scale,
        )
    raise TypeError(f"Unsupported loss options: {type(options).__name__}")


def _get_parameter_groups(
    named_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
    options: AdamWOptions,
) -> list[dict]:
    """Group parameters by the first regex matching their owning module name."""
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


register_stage(LoopedMNLIStageOptions, _build_stage_template)
