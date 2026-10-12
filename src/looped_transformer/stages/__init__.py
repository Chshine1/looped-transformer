from looped_transformer.config.data import DataOptions
from looped_transformer.config.training import TrainingOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.stages import looped_mnli
from looped_transformer.stages.registry import build_stage_template
from looped_transformer.types.train import StageTemplate


def ensure_builtin_stages_registered() -> None:
    """Import built-in stage modules for their one-time registrations."""
    assert looped_mnli is not None


def build_stage_templates(
    options: TrainingOptions,
    data_reader: MNLIDataReader,
    data_options: DataOptions,
) -> list[StageTemplate]:
    return [
        build_stage_template(
            template_id,
            options.stage_templates[template_id],
            data_reader,
            data_options,
        )
        for template_id in options.stages
    ]


__all__ = ["build_stage_templates", "ensure_builtin_stages_registered"]
