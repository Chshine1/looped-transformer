from lagom import Container, Singleton
from lagom.interfaces import ReadableContainer

from looped_transformer.app.config import AppOptions
from looped_transformer.core.artifacts import ArtifactSpec, get_file_fingerprint
from looped_transformer.core.checkpointing import CheckpointLoader, CheckpointSaver
from looped_transformer.core.checkpointing.saver import CheckpointSavingOptions
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.core.training.config import (
    ReproducibilityOptions,
    TrainingRuntimeOptions,
)
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.core.training.stage import TrainingOptions
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.clutrr.data_provider import CLUTRRDataOptions
from looped_transformer.domains.clutrr.stage import (
    register_looped_clutrr_stage_template,
)
from looped_transformer.domains.clutrr.types import CLUTRRBatch
from looped_transformer.domains.mnli.evaluation.evaluation import EvaluationOptions
from looped_transformer.domains.mnli.services.data_provider import MNLIDataOptions
from looped_transformer.domains.mnli.training.stage import (
    register_looped_mnli_stage_template,
)
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.text_classification.config.options import LoopOptions
from looped_transformer.domains.text_classification.loop_diagnostics import (
    LoopDiagnosticsOptions,
)
from looped_transformer.domains.text_classification.loss import (
    BackboneLossInput,
)
from looped_transformer.domains.text_classification.model import ModelOptions
from looped_transformer.domains.text_classification.tokenizer import (
    PairTokenizer,
    TokenizerOptions,
)
from looped_transformer.domains.text_classification.types import ModelForwardOutput


def build_container(options: AppOptions) -> Container:
    container = Container()
    container[AppOptions] = options
    artifact_values = {
        "runtime": options.runtime,
        "model": options.model,
        "tokenizer": options.tokenizer,
        "stage": options.training.stage,
        "resume_from": _checkpoint_reference(options.training.resume_from),
    }
    container[ArtifactSpec] = ArtifactSpec(
        format="looped-transformer-training",
        values=artifact_values,
    )
    container[CheckpointSavingOptions] = options.checkpoint
    container[TokenizerOptions] = options.tokenizer
    container[SequenceOptions] = options.tokenizer.sequence
    container[ModelOptions] = options.model
    container[LoopOptions] = options.model.loop
    container[DeviceOptions] = options.runtime.device
    container[PrecisionOptions] = options.runtime.precision
    container[ReproducibilityOptions] = options.runtime.reproducibility
    container[TrainingRuntimeOptions] = options.runtime.training
    container[TrainingOptions] = options.training
    stage_registry = TrainingStageRegistry[
        MNLIBatch, ModelForwardOutput, BackboneLossInput
    ](options.training)
    register_looped_mnli_stage_template(stage_registry)
    container[
        TrainingStageRegistry[MNLIBatch, ModelForwardOutput, BackboneLossInput]
    ] = stage_registry
    clutrr_stage_registry = TrainingStageRegistry[
        CLUTRRBatch, ModelForwardOutput, BackboneLossInput
    ](options.training)
    register_looped_clutrr_stage_template(clutrr_stage_registry)
    container[
        TrainingStageRegistry[CLUTRRBatch, ModelForwardOutput, BackboneLossInput]
    ] = clutrr_stage_registry
    container[CheckpointLoader] = CheckpointLoader
    container[CheckpointSaver] = CheckpointSaver
    container[PairTokenizer] = Singleton(_build_tokenizer)
    container[ModelTrainerFactory] = Singleton(ModelTrainerFactory)
    return container


def configure_evaluation(container: Container, options: EvaluationOptions) -> Container:
    evaluation_container = container.clone()
    evaluation_container[EvaluationCheckpointOptions] = options.checkpoint
    evaluation_container[EvaluationOutputOptions] = options.output
    evaluation_container[LoopDiagnosticsOptions] = options.diagnostics
    evaluation_container[MNLIDataOptions] = options.data
    evaluation_container[StructuralLogger] = Singleton(StructuralLogger)
    evaluation_container[EvaluatorFactory] = Singleton(EvaluatorFactory)
    return evaluation_container


def configure_clutrr_evaluation(
    container: Container,
    checkpoint: EvaluationCheckpointOptions,
    output: EvaluationOutputOptions,
    diagnostics: LoopDiagnosticsOptions,
    data: CLUTRRDataOptions,
) -> Container:
    evaluation_container = container.clone()
    evaluation_container[EvaluationCheckpointOptions] = checkpoint
    evaluation_container[EvaluationOutputOptions] = output
    evaluation_container[LoopDiagnosticsOptions] = diagnostics
    evaluation_container[CLUTRRDataOptions] = data
    evaluation_container[StructuralLogger] = Singleton(StructuralLogger)
    evaluation_container[EvaluatorFactory] = Singleton(EvaluatorFactory)
    return evaluation_container


def _build_tokenizer(container: ReadableContainer) -> PairTokenizer:
    return PairTokenizer(container[TokenizerOptions])


def _checkpoint_reference(path) -> dict | None:
    if path is None:
        return None
    return {
        "path": str(path),
        "fingerprint": get_file_fingerprint(path),
    }
