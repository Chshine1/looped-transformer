import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, ClassVar, Self

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.config.data import DataOptions
from looped_transformer.config.training import (
    CommonStageOptions,
    StageOptions,
    check_keys,
    require_mapping,
)
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.loss import (
    AdaptiveJointLoss,
    DepthWeightedCrossEntropyLoss,
    LoopedMNLILoss,
    LastLoopCrossEntropyLoss,
    OracleHaltingLoss,
)
from looped_transformer.stages.registry import register_stage
from looped_transformer.types.data import DataSplit
from looped_transformer.types.train import StageTemplate


@dataclass(frozen=True, slots=True)
class AdamWOptions:
    learning_rates: dict[str, float]
    weight_decay: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    def __post_init__(self) -> None:
        if not self.learning_rates:
            raise ValueError("learning_rates must not be empty")
        for pattern, learning_rate in self.learning_rates.items():
            re.compile(pattern)
            if learning_rate < 0:
                raise ValueError("learning rates must not be negative")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must not be negative")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW beta values must be in [0, 1)")
        if self.epsilon <= 0:
            raise ValueError("AdamW epsilon must be positive")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "AdamW options")
        check_keys(
            raw,
            {"learning_rates", "weight_decay"},
            {"beta1", "beta2", "epsilon"},
        )
        learning_rates = require_mapping(raw["learning_rates"], "learning_rates")
        if not all(
            isinstance(pattern, str) and isinstance(rate, int | float)
            for pattern, rate in learning_rates.items()
        ):
            raise TypeError("learning_rates must map regex strings to numbers")
        return cls(
            learning_rates=dict(learning_rates),
            weight_decay=raw["weight_decay"],
            beta1=raw.get("beta1", 0.9),
            beta2=raw.get("beta2", 0.999),
            epsilon=raw.get("epsilon", 1e-8),
        )


@dataclass(frozen=True, slots=True)
class WarmupCosineOptions:
    warmup_ratio: float
    warmup_start_factor: float
    minimum_learning_rate_factor: float

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 < self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be in (0, 1]")
        if not 0 <= self.minimum_learning_rate_factor <= 1:
            raise ValueError("minimum_learning_rate_factor must be in [0, 1]")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "warmup cosine options")
        check_keys(
            raw,
            {
                "warmup_ratio",
                "warmup_start_factor",
                "minimum_learning_rate_factor",
            },
            {"type"},
        )
        if raw.get("type", "warmup_cosine") != "warmup_cosine":
            raise ValueError("Expected scheduler type 'warmup_cosine'")
        return cls(**raw)


@dataclass(frozen=True, slots=True)
class WarmupConstantOptions:
    warmup_ratio: float
    warmup_start_factor: float

    def __post_init__(self) -> None:
        if not 0 <= self.warmup_ratio < 1:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if not 0 < self.warmup_start_factor <= 1:
            raise ValueError("warmup_start_factor must be in (0, 1]")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "warmup constant options")
        check_keys(raw, {"type", "warmup_ratio", "warmup_start_factor"})
        if raw["type"] != "warmup_constant":
            raise ValueError("Expected scheduler type 'warmup_constant'")
        return cls(
            warmup_ratio=raw["warmup_ratio"],
            warmup_start_factor=raw["warmup_start_factor"],
        )


SchedulerOptions = WarmupCosineOptions | WarmupConstantOptions


def _load_scheduler_options(value: Any) -> SchedulerOptions:
    raw = require_mapping(value, "scheduler options")
    scheduler_type = raw.get("type", "warmup_cosine")
    if scheduler_type == "warmup_cosine":
        values = dict(raw)
        values.pop("type", None)
        return WarmupCosineOptions.model_validate(values)
    if scheduler_type == "warmup_constant":
        return WarmupConstantOptions.model_validate(raw)
    raise ValueError(f"Unknown scheduler type: {scheduler_type!r}")


@dataclass(frozen=True, slots=True)
class LoopedMNLILossOptions:
    expected_halting_time: int
    halting_loss_scale: float
    auxiliary_loss_scale: float

    def __post_init__(self) -> None:
        if self.expected_halting_time <= 0:
            raise ValueError("expected_halting_time must be positive")
        if self.halting_loss_scale < 0 or self.auxiliary_loss_scale < 0:
            raise ValueError("loss scales must not be negative")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "looped MNLI loss options")
        check_keys(
            raw,
            {
                "expected_halting_time",
                "halting_loss_scale",
                "auxiliary_loss_scale",
            },
            {"type"},
        )
        if raw.get("type", "ponder") != "ponder":
            raise ValueError("Expected loss type 'ponder'")
        return cls(
            expected_halting_time=raw["expected_halting_time"],
            halting_loss_scale=raw["halting_loss_scale"],
            auxiliary_loss_scale=raw["auxiliary_loss_scale"],
        )


@dataclass(frozen=True, slots=True)
class LastLoopLossOptions:
    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "last-loop loss options")
        check_keys(raw, {"type"})
        if raw["type"] != "last_loop_cross_entropy":
            raise ValueError("Expected loss type 'last_loop_cross_entropy'")
        return cls()


@dataclass(frozen=True, slots=True)
class DepthWeightedLossOptions:
    depth_weight_growth: float
    monotonic_loss_scale: float

    def __post_init__(self) -> None:
        if self.depth_weight_growth < 0 or self.monotonic_loss_scale < 0:
            raise ValueError("Depth-loss parameters must not be negative")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "depth-weighted loss options")
        check_keys(raw, {"type", "depth_weight_growth", "monotonic_loss_scale"})
        if raw["type"] != "depth_weighted_cross_entropy":
            raise ValueError("Expected loss type 'depth_weighted_cross_entropy'")
        return cls(
            depth_weight_growth=raw["depth_weight_growth"],
            monotonic_loss_scale=raw["monotonic_loss_scale"],
        )


@dataclass(frozen=True, slots=True)
class OracleHaltingLossOptions:
    compute_cost: float
    target_temperature: float

    def __post_init__(self) -> None:
        if self.compute_cost < 0:
            raise ValueError("compute_cost must not be negative")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "oracle-halting loss options")
        check_keys(raw, {"type", "compute_cost", "target_temperature"})
        if raw["type"] != "oracle_halting":
            raise ValueError("Expected loss type 'oracle_halting'")
        return cls(
            compute_cost=raw["compute_cost"],
            target_temperature=raw["target_temperature"],
        )


@dataclass(frozen=True, slots=True)
class AdaptiveJointLossOptions:
    compute_cost: float
    target_temperature: float
    controller_loss_scale: float
    depth_loss_scale: float
    depth_weight_growth: float
    monotonic_loss_scale: float

    def __post_init__(self) -> None:
        nonnegative = (
            self.compute_cost,
            self.controller_loss_scale,
            self.depth_loss_scale,
            self.depth_weight_growth,
            self.monotonic_loss_scale,
        )
        if any(value < 0 for value in nonnegative):
            raise ValueError("Joint-loss scales must not be negative")
        if self.target_temperature <= 0:
            raise ValueError("target_temperature must be positive")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "adaptive-joint loss options")
        keys = {
            "type",
            "compute_cost",
            "target_temperature",
            "controller_loss_scale",
            "depth_loss_scale",
            "depth_weight_growth",
            "monotonic_loss_scale",
        }
        check_keys(raw, keys)
        if raw["type"] != "adaptive_joint":
            raise ValueError("Expected loss type 'adaptive_joint'")
        values = dict(raw)
        values.pop("type")
        return cls(**values)


LossOptions = (
        LoopedMNLILossOptions
        | LastLoopLossOptions
        | DepthWeightedLossOptions
        | OracleHaltingLossOptions
        | AdaptiveJointLossOptions
)


def _load_loss_options(value: Any) -> LossOptions:
    raw = require_mapping(value, "loss options")
    loss_type = raw.get("type", "ponder")
    option_types = {
        "ponder": LoopedMNLILossOptions,
        "last_loop_cross_entropy": LastLoopLossOptions,
        "depth_weighted_cross_entropy": DepthWeightedLossOptions,
        "oracle_halting": OracleHaltingLossOptions,
        "adaptive_joint": AdaptiveJointLossOptions,
    }
    options_type = option_types.get(loss_type)
    if options_type is None:
        raise ValueError(f"Unknown loss type: {loss_type!r}")
    return options_type.model_validate(raw)


@dataclass(frozen=True, slots=True)
class LoopedMNLIStageOptions(StageOptions):
    stage_type: ClassVar[str] = "looped_mnli"

    common: CommonStageOptions
    optimizer: AdamWOptions
    scheduler: SchedulerOptions
    loss: LossOptions
    type: str = field(init=False, default=stage_type)

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "looped MNLI stage options")
        check_keys(raw, {"type", "common", "optimizer", "scheduler", "loss"})
        if raw["type"] != cls.stage_type:
            raise ValueError(f"Expected stage type {cls.stage_type!r}")
        return cls(
            common=CommonStageOptions.model_validate(raw["common"]),
            optimizer=AdamWOptions.model_validate(raw["optimizer"]),
            scheduler=_load_scheduler_options(raw["scheduler"]),
            loss=_load_loss_options(raw["loss"]),
        )


def _build_stage_template(
    template_id: str,
    options: StageOptions,
    data_reader: MNLIDataReader,
    data_options: DataOptions,
) -> StageTemplate:
    if not isinstance(options, LoopedMNLIStageOptions):
        raise TypeError("looped_mnli requires LoopedMNLIStageOptions")

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
        row_count = data_reader.count_examples(DataSplit.TRAIN)
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
    if isinstance(options, LoopedMNLILossOptions):
        return LoopedMNLILoss(
            expected_time=options.expected_halting_time,
            halt_scale=options.halting_loss_scale,
            aux_scale=options.auxiliary_loss_scale,
        )
    if isinstance(options, LastLoopLossOptions):
        return LastLoopCrossEntropyLoss()
    if isinstance(options, DepthWeightedLossOptions):
        return DepthWeightedCrossEntropyLoss(
            depth_weight_growth=options.depth_weight_growth,
            monotonic_loss_scale=options.monotonic_loss_scale,
        )
    if isinstance(options, OracleHaltingLossOptions):
        return OracleHaltingLoss(
            compute_cost=options.compute_cost,
            target_temperature=options.target_temperature,
        )
    if isinstance(options, AdaptiveJointLossOptions):
        return AdaptiveJointLoss(
            compute_cost=options.compute_cost,
            target_temperature=options.target_temperature,
            controller_loss_scale=options.controller_loss_scale,
            depth_loss_scale=options.depth_loss_scale,
            depth_weight_growth=options.depth_weight_growth,
            monotonic_loss_scale=options.monotonic_loss_scale,
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
