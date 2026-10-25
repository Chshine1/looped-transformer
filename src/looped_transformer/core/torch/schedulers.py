"""Configuration extensions for PyTorch learning-rate schedulers."""

from typing import Annotated, Literal

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
