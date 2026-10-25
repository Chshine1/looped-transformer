import math
from typing import Annotated, Literal

import torch
from pydantic import Field

from looped_transformer.core.config.base import (
    ExclusiveProbability,
    OptionsModel,
    Probability,
)


class WarmupCosineOptions(OptionsModel):
    type: Literal["warmup_cosine"]
    warmup_ratio: ExclusiveProbability
    warmup_start_factor: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]
    minimum_learning_rate_factor: Probability


class WarmupConstantOptions(OptionsModel):
    type: Literal["warmup_constant"]
    warmup_ratio: ExclusiveProbability
    warmup_start_factor: Annotated[float, Field(gt=0, le=1, allow_inf_nan=False)]


SchedulerOptions = Annotated[
    WarmupCosineOptions | WarmupConstantOptions, Field(discriminator="type")
]


def create_scheduler(
    optimizer: torch.optim.Optimizer, options: SchedulerOptions, total_steps: int
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup_steps = max(1, int(options.warmup_ratio * total_steps))
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=options.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    if isinstance(options, WarmupCosineOptions):
        cosine_steps = max(1, total_steps - warmup_steps)
        minimum_factor = options.minimum_learning_rate_factor

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
