from pathlib import Path
from typing import Any

import yaml

from looped_transformer.config.base import OptionsModel
from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.runtime import RuntimeOptions


class AppOptions(OptionsModel):
    runtime: RuntimeOptions
    data: DataOptions
    model: ModelOptions


def load_options(path: str | Path) -> AppOptions:
    from looped_transformer.stages import ensure_builtin_stages_registered

    ensure_builtin_stages_registered()
    with Path(path).open(encoding="utf-8") as stream:
        raw: dict[str, Any] = yaml.safe_load(stream)

    return AppOptions.model_validate(raw)
