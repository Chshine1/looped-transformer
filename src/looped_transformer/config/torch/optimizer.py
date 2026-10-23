import re

from pydantic import Field
from pydantic import field_validator

from looped_transformer.config import OptionsModel
from looped_transformer.config.base import ExclusiveProbability
from looped_transformer.config.base import NonNegativeFloat
from looped_transformer.config.base import PositiveFloat


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
