import re
from collections.abc import Iterable

import torch
from pydantic import Field, field_validator
from torch import nn

from looped_transformer.core.config.base import (
    ExclusiveProbability,
    NonNegativeFloat,
    OptionsModel,
    PositiveFloat,
)


class AdamWOptions(OptionsModel):
    learning_rates: dict[str, NonNegativeFloat] = Field(min_length=1)
    weight_decay: NonNegativeFloat
    beta1: ExclusiveProbability = 0.9
    beta2: ExclusiveProbability = 0.999
    epsilon: PositiveFloat = 1e-8

    # noinspection nested-decorators
    @field_validator("learning_rates")
    @classmethod
    def compile_module_patterns(cls, value: dict[str, float]) -> dict[str, float]:
        for pattern in value:
            re.compile(pattern)
        return value


def create_adam_w_optimizer(
    options: AdamWOptions,
    target_parameters: Iterable[tuple[str, nn.parameter.Parameter]],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        _get_parameter_groups(target_parameters, options),
        betas=(options.beta1, options.beta2),
        eps=options.epsilon,
        weight_decay=options.weight_decay,
    )


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
