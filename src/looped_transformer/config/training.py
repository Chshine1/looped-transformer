import re
from typing import Any, ClassVar

from pydantic import Field, SerializeAsAny, field_validator, model_validator

from looped_transformer.config import DataOptions
from looped_transformer.config.base import (
    NonEmptyString,
    OptionsModel,
    PositiveFloat,
    PositiveInt,
)


class CommonStageOptions(OptionsModel):
    epochs: PositiveInt
    frozen_patterns: str = r"(?!)"
    evaluation_patterns: str = r"(?!)"
    max_gradient_norm: PositiveFloat = 1.0
    gradient_accumulation_steps: PositiveInt = 1
    save_every_epochs: PositiveInt = 1
    data: DataOptions

    @field_validator("frozen_patterns", "evaluation_patterns")
    @classmethod
    def compile_regex(cls, value: str) -> str:
        re.compile(value)
        return value


class StageOptions(OptionsModel):
    stage_type: ClassVar[str]


class TrainingOptions(OptionsModel):
    stage_templates: dict[NonEmptyString, SerializeAsAny[StageOptions]]
    stages: list[NonEmptyString] = Field(min_length=1)
    log_every_steps: PositiveInt = 50

    @field_validator("stage_templates", mode="before")
    @classmethod
    def load_registered_stage_options(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        from looped_transformer.stages.registry import load_stage_options

        return {
            template_id: load_stage_options(options)
            for template_id, options in value.items()
        }

    @model_validator(mode="after")
    def validate_stage_sequence(self) -> "TrainingOptions":
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("a stage template can only appear once in a run")
        missing = set(self.stages) - self.stage_templates.keys()
        if missing:
            raise ValueError(
                "stages reference missing templates: " + ", ".join(sorted(missing))
            )
        return self
