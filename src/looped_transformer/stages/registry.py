from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from looped_transformer.abstractions.training import StageTemplate
from looped_transformer.config.training import StageOptions

StageBuilder = Callable[[str, StageOptions], StageTemplate]


@dataclass(frozen=True, slots=True)
class StageDefinition:
    options_model: type[StageOptions]
    builder: StageBuilder


_STAGE_DEFINITIONS: dict[str, StageDefinition] = {}


def register_stage(options_model: type[StageOptions], builder: StageBuilder) -> None:
    stage_type = options_model.stage_type
    if stage_type in _STAGE_DEFINITIONS:
        raise ValueError(f"Stage type is already registered: {stage_type!r}")
    _STAGE_DEFINITIONS[stage_type] = StageDefinition(options_model, builder)


def load_stage_options(value: Any) -> StageOptions:
    if not isinstance(value, Mapping):
        raise TypeError("stage template must be a mapping")
    raw: Mapping[str, Any] = value
    stage_type = raw.get("type")
    if not isinstance(stage_type, str) or not stage_type:
        raise ValueError("A stage template requires a non-empty 'type'")
    try:
        definition = _STAGE_DEFINITIONS[stage_type]
    except KeyError:
        available = ", ".join(sorted(_STAGE_DEFINITIONS)) or "<none>"
        raise ValueError(
            f"Unknown stage type {stage_type!r}; registered types: {available}"
        ) from None
    return definition.options_model.model_validate(raw)


def build_stage_template(
    template_id: str,
    options: StageOptions,
) -> StageTemplate:
    try:
        definition = _STAGE_DEFINITIONS[options.stage_type]
    except KeyError:
        raise ValueError(f"Unregistered stage type: {options.stage_type!r}") from None
    if not isinstance(options, definition.options_model):
        raise TypeError(
            f"Options for {options.stage_type!r} must be "
            f"{definition.options_model.__name__}"
        )
    return definition.builder(template_id, options)
