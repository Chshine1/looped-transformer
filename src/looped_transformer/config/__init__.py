from looped_transformer.config.data import CommonDataOptions
from typing import cast
from pathlib import Path
from typing import Any

from hydra import initialize_config_dir, compose
from omegaconf import DictConfig, OmegaConf

from looped_transformer.config.base import OptionsModel
from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.runtime import RuntimeOptions


class AppOptions(OptionsModel):
    runtime: RuntimeOptions
    data: CommonDataOptions
    model: ModelOptions


def load_options(
    config_dir: str,
    overrides: list[str],
) -> AppOptions:
    from looped_transformer.stages import ensure_builtin_stages_registered

    ensure_builtin_stages_registered()

    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(config_dir).resolve()),
    ):
        cfg: DictConfig = compose(
            config_name="config",
            overrides=overrides,
        )

    raw: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(cfg, resolve=True)
    )
    return AppOptions.model_validate(raw)
