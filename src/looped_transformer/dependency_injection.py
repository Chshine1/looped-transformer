from lagom import Container, Singleton
from lagom.interfaces import ReadableContainer

from looped_transformer.artifacts import RunArtifactOptions
from looped_transformer.checkpoint_loader import CheckpointLoader
from looped_transformer.checkpoint_saver import CheckpointSaver, CheckpointSavingOptions
from looped_transformer.config import AppOptions
from looped_transformer.config.evaluation import EvaluationOptions
from looped_transformer.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.config.sequence import SequenceOptions
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.core.training.config import (
    ReproducibilityOptions,
    TrainingRuntimeOptions,
)
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.data_reader import DataOptions
from looped_transformer.modules.config import LoopOptions
from looped_transformer.modules.controller_teacher import ControllerTeacherOptions
from looped_transformer.modules.model import ModelOptions
from looped_transformer.policies.loop_diagnostics import LoopDiagnosticsOptions
from looped_transformer.tokenizer import (
    MNLITokenizer,
    TokenizerOptions,
)


def build_container(options: AppOptions) -> Container:
    container = Container()
    container[AppOptions] = options
    container[RunArtifactOptions] = RunArtifactOptions(
        runtime=options.runtime,
        model=options.model,
        tokenizer=options.tokenizer,
        training=options.training,
    )
    container[CheckpointSavingOptions] = options.checkpoint
    container[TokenizerOptions] = options.tokenizer
    container[SequenceOptions] = options.tokenizer.sequence
    container[ModelOptions] = options.model
    container[LoopOptions] = options.model.loop
    container[ControllerTeacherOptions] = ControllerTeacherOptions(
        model_dimension=options.model.hidden_state_dimension,
        ffn_dimension=options.model.transformer.gelu_ffn_dimension,
        heads_count=options.model.transformer.attention_heads_count,
        dropout=options.model.dropout.controller,
    )
    container[DeviceOptions] = options.runtime.device
    container[PrecisionOptions] = options.runtime.precision
    container[ReproducibilityOptions] = options.runtime.reproducibility
    container[TrainingRuntimeOptions] = options.runtime.training
    container[CheckpointLoader] = CheckpointLoader
    container[CheckpointSaver] = CheckpointSaver
    container[MNLITokenizer] = Singleton(_build_tokenizer)
    container[ModelTrainerFactory] = Singleton(ModelTrainerFactory)
    return container


def configure_evaluation(container: Container, options: EvaluationOptions) -> Container:
    evaluation_container = container.clone()
    evaluation_container[EvaluationCheckpointOptions] = options.checkpoint
    evaluation_container[EvaluationOutputOptions] = options.output
    evaluation_container[LoopDiagnosticsOptions] = options.diagnostics
    evaluation_container[DataOptions] = options.data
    evaluation_container[StructuralLogger] = Singleton(StructuralLogger)
    evaluation_container[EvaluatorFactory] = Singleton(EvaluatorFactory)
    return evaluation_container


def _build_tokenizer(container: ReadableContainer) -> MNLITokenizer:
    return MNLITokenizer(container[TokenizerOptions])
