from looped_transformer.abstractions.training import StageTemplate
from looped_transformer.stages import looped_mnli
from looped_transformer.stages.config import TrainingOptions
from looped_transformer.stages.registry import build_stage_template


def ensure_builtin_stages_registered() -> None:
    assert looped_mnli is not None


def build_stage_templates(
    options: TrainingOptions,
) -> list[StageTemplate]:
    return [
        build_stage_template(
            template_id,
            options.stage_templates[template_id],
        )
        for template_id in options.stages
    ]


__all__ = ["build_stage_templates", "ensure_builtin_stages_registered"]
