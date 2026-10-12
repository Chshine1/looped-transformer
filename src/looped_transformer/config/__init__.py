from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import (
    EvaluationOptions,
    HaltingOptions,
    ModelOptions,
    TransformerOptions,
)
from looped_transformer.config.runtime import RuntimeOptions
from looped_transformer.config.training import TrainingOptions


@dataclass(frozen=True, slots=True)
class AppOptions:
    runtime: RuntimeOptions
    data: DataOptions
    model: ModelOptions


def load_options(path: str | Path) -> AppOptions:
    from looped_transformer.stages import ensure_builtin_stages_registered

    ensure_builtin_stages_registered()
    with Path(path).open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)

    model_options = raw["model"]
    return AppOptions(
        runtime=RuntimeOptions(**raw["runtime"]),
        data=DataOptions(**raw["data"]),
        model=ModelOptions(
            hidden_state_dimension=model_options["hidden_state_dimension"],
            output_classes_count=model_options["output_classes_count"],
            max_iterations=model_options["max_iterations"],
            max_sequence_length=model_options["max_sequence_length"],
            dropout=model_options["dropout"],
            transformer=TransformerOptions(**model_options["transformer"]),
            halting=HaltingOptions(**model_options["halting"]),
            training=TrainingOptions.model_validate(model_options["training"]),
            evaluation=EvaluationOptions(**model_options["evaluation"]),
        ),
    )
