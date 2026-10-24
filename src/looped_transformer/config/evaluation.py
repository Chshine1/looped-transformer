from pydantic import Field

from looped_transformer.config.base import OptionsModel
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.data_reader import DataOptions
from looped_transformer.policies.loop_diagnostics import LoopDiagnosticsOptions


class EvaluationOptions(OptionsModel):
    checkpoint: EvaluationCheckpointOptions
    data: DataOptions
    output: EvaluationOutputOptions
    diagnostics: LoopDiagnosticsOptions = Field(default_factory=LoopDiagnosticsOptions)
