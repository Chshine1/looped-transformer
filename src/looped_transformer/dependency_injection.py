from lagom import Container, Singleton
from lagom.interfaces import ReadableContainer

from looped_transformer.checkpoint_manager import CheckpointLoader, CheckpointSaver
from looped_transformer.config import (
    AppOptions,
    CommonDataOptions,
    DataOptions,
    ModelOptions,
    RuntimeOptions,
)
from looped_transformer.config.evaluation import EvaluationOptions
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.tokenizer import MNLITokenizer


def build_container(options: AppOptions) -> Container:
    container = Container()
    container[AppOptions] = options
    container[CommonDataOptions] = options.data
    container[ModelOptions] = options.model
    container[RuntimeOptions] = options.runtime
    container[CheckpointLoader] = CheckpointLoader
    container[CheckpointSaver] = CheckpointSaver
    container[MNLITokenizer] = Singleton(_build_tokenizer)
    container[ModelTrainerFactory] = Singleton(ModelTrainerFactory)
    return container


def configure_evaluation(container: Container, options: EvaluationOptions) -> Container:
    evaluation_container = container.clone()
    evaluation_container[EvaluationOptions] = options
    evaluation_container[DataOptions] = options.data
    evaluation_container[StructuralLogger] = Singleton(_build_logger)
    evaluation_container[EvaluatorFactory] = Singleton(EvaluatorFactory)
    return evaluation_container


def _build_tokenizer(container: ReadableContainer) -> MNLITokenizer:
    return MNLITokenizer(container[ModelOptions].max_sequence_length)


def _build_logger(
    container: ReadableContainer,
) -> StructuralLogger:
    return StructuralLogger(container[EvaluationOptions].output_dir)
