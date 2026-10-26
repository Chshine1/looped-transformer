import re
from typing import ClassVar

from pydantic import JsonValue, field_validator

from looped_transformer.core.config.base import (
    ConfigPath,
    NonEmptyString,
    OptionsModel,
    PositiveFloat,
    PositiveInt,
)


class StageOptions(OptionsModel):
    epochs: PositiveInt
    frozen_patterns: str = r"(?!)"
    evaluation_patterns: str = r"(?!)"
    max_gradient_norm: PositiveFloat = 1.0
    gradient_accumulation_steps: PositiveInt = 1
    save_every_epochs: PositiveInt = 1

    # noinspection nested-decorators
    @field_validator("frozen_patterns", "evaluation_patterns")
    @classmethod
    def compile_regex(cls, value: str) -> str:
        re.compile(value)
        return value


class StageTemplate(OptionsModel):
    type: ClassVar[str]
    id: NonEmptyString
    options: StageOptions


class TrainingOptions(OptionsModel):
    stage: dict[str, JsonValue]
    resume_from: ConfigPath | None = None
