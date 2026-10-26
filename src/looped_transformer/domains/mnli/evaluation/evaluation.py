from looped_transformer.core.config.base import OptionsModel
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.domains.mnli.evaluation.loop_diagnostics import (
    LoopDiagnosticsOptions,
)
from looped_transformer.domains.mnli.services.data_provider import MNLIDataOptions


class EvaluationOptions(OptionsModel):
    checkpoint: EvaluationCheckpointOptions
    data: MNLIDataOptions
    output: EvaluationOutputOptions
    diagnostics: LoopDiagnosticsOptions
