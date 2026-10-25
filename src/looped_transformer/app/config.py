from pathlib import Path
from typing import Any, cast

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from looped_transformer.core.checkpointing.saver import CheckpointSavingOptions
from looped_transformer.core.config.base import OptionsModel
from looped_transformer.core.config.runtime import RuntimeOptions
from looped_transformer.core.training.stage import TrainingOptions
from looped_transformer.domains.mnli.model import ModelOptions
from looped_transformer.domains.mnli.services.tokenizer import TokenizerOptions


class AppOptions(OptionsModel):
    runtime: RuntimeOptions
    checkpoint: CheckpointSavingOptions
    tokenizer: TokenizerOptions
    model: ModelOptions
    training: TrainingOptions


def load_options(
    config_dir: str,
    overrides: list[str],
) -> AppOptions:
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(config_dir).resolve()),
    ):
        config: DictConfig = compose(
            config_name="config",
            overrides=overrides,
        )

    raw: dict[str, Any] = cast(
        dict[str, Any], OmegaConf.to_container(config, resolve=True)
    )
    return AppOptions.model_validate(raw)
