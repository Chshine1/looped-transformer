from looped_transformer.config.base import Probability
from pydantic import Field
from typing import Annotated
from looped_transformer.config.base import ExclusiveProbability
from typing import Literal
from looped_transformer.config import OptionsModel


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
