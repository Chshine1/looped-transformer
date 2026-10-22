from lagom import Alias, Container, Singleton
from lagom.interfaces import ReadableContainer

from looped_transformer.checkpoint_manager import CheckpointLoader, CheckpointSaver
from looped_transformer.config import (
    AppOptions,
    CommonDataOptions,
    DataOptions,
    ModelOptions,
    RuntimeOptions,
)
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.evaluation import Evaluator, StructuredEvaluationLogger
from looped_transformer.evaluation.options import EvaluationOptions
from looped_transformer.evaluation.policies import LoopDiagnosticsPolicy
from looped_transformer.evaluation.policy import EvaluationPolicy
from looped_transformer.runner import MNLIRunner
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
    container[MNLIRunner] = MNLIRunner
    return container


def configure_evaluation(container: Container, options: EvaluationOptions) -> Container:
    evaluation_container = container.clone()
    evaluation_container[EvaluationOptions] = options
    evaluation_container[DataOptions] = options.data
    evaluation_container[MNLIDataReader] = MNLIDataReader
    evaluation_container[StructuredEvaluationLogger] = Singleton(
        _build_evaluation_logger
    )
    evaluation_container[LoopDiagnosticsPolicy] = Singleton(
        _build_loop_diagnostics_policy
    )
    evaluation_container[EvaluationPolicy] = Alias(LoopDiagnosticsPolicy)
    evaluation_container[Evaluator] = Evaluator
    return evaluation_container


def _build_tokenizer(container: ReadableContainer) -> MNLITokenizer:
    return MNLITokenizer(container[ModelOptions].max_sequence_length)


def _build_evaluation_logger(
    container: ReadableContainer,
) -> StructuredEvaluationLogger:
    return StructuredEvaluationLogger(container[EvaluationOptions].output_dir)


def _build_loop_diagnostics_policy(
    container: ReadableContainer,
) -> LoopDiagnosticsPolicy:
    model = container[ModelOptions]
    runtime = container[RuntimeOptions]
    options = container[EvaluationOptions]
    return LoopDiagnosticsPolicy(
        max_iterations=model.max_iterations,
        device=runtime.device,
        label_smoothing=options.label_smoothing,
        primary_reduction=options.primary_reduction,
        shuffle_oracle_repeats=options.shuffle_oracle_repeats,
        logger=container[StructuredEvaluationLogger],
    )
