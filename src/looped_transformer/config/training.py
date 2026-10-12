import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Set
from dataclasses import dataclass
from typing import Any, ClassVar, Self


def check_keys(
    raw: Mapping[str, Any], required: Set[str], optional: Set[str] = frozenset()
) -> None:
    missing = required - raw.keys()
    if missing:
        raise ValueError(f"Missing options: {', '.join(sorted(missing))}")
    unexpected = raw.keys() - required - optional
    if unexpected:
        raise ValueError(f"Unexpected options: {', '.join(sorted(unexpected))}")


def require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


@dataclass(frozen=True, slots=True)
class CommonStageOptions:
    epochs: int
    frozen_patterns: str = r"(?!)"
    max_gradient_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    save_every_epochs: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.epochs, bool) or not isinstance(self.epochs, int):
            raise TypeError("epochs must be an integer")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.save_every_epochs <= 0:
            raise ValueError("save_every_epochs must be positive")
        re.compile(self.frozen_patterns)

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "common stage options")
        check_keys(
            raw,
            {"epochs"},
            {
                "frozen_patterns",
                "max_gradient_norm",
                "gradient_accumulation_steps",
                "save_every_epochs",
            },
        )
        return cls(
            epochs=raw["epochs"],
            frozen_patterns=raw.get("frozen_patterns", r"(?!)"),
            max_gradient_norm=raw.get("max_gradient_norm", 1.0),
            gradient_accumulation_steps=raw.get("gradient_accumulation_steps", 1),
            save_every_epochs=raw.get("save_every_epochs", 1),
        )


class StageOptions(ABC):
    stage_type: ClassVar[str]

    @classmethod
    @abstractmethod
    def model_validate(cls, value: Any) -> Self:
        """Construct and validate this concrete stage's complete configuration."""


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    stage_templates: dict[str, StageOptions]
    stages: list[str]
    log_every_steps: int = 50

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("At least one training stage must be configured")
        if len(self.stages) != len(set(self.stages)):
            raise ValueError("A stage template can only appear once in a run")
        missing = set(self.stages) - self.stage_templates.keys()
        if missing:
            raise ValueError(
                "Stages reference missing templates: " + ", ".join(sorted(missing))
            )
        if self.log_every_steps <= 0:
            raise ValueError("log_every_steps must be positive")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        raw = require_mapping(value, "training options")
        check_keys(raw, {"stage_templates", "stages"}, {"log_every_steps"})

        templates_raw = require_mapping(raw["stage_templates"], "stage_templates")
        from looped_transformer.stages.registry import load_stage_options

        templates: dict[str, StageOptions] = {}
        for template_id, template_raw in templates_raw.items():
            if not isinstance(template_id, str) or not template_id:
                raise ValueError("Stage template ids must be non-empty strings")
            templates[template_id] = load_stage_options(template_raw)

        stages = raw["stages"]
        if not isinstance(stages, list) or not all(
            isinstance(stage_id, str) and stage_id for stage_id in stages
        ):
            raise TypeError("stages must be a list of non-empty template ids")

        return cls(
            stage_templates=templates,
            stages=stages,
            log_every_steps=raw.get("log_every_steps", 50),
        )
