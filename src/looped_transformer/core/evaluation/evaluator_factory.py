from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training import (
    ModelTrainingCollator,
    IDataProvider,
)
from looped_transformer.checkpoint_manager import CheckpointLoader
from looped_transformer.config import RuntimeOptions
from looped_transformer.config.evaluation import EvaluationOptions
from looped_transformer.core.evaluation import Evaluator
from looped_transformer.core.evaluation.policy import EvaluationPolicy


class EvaluatorFactory:
    def __init__(
        self,
        checkpoint_loader: CheckpointLoader,
        runtime_options: RuntimeOptions,
        evaluation_options: EvaluationOptions,
    ) -> None:
        self._checkpoint_loader = checkpoint_loader
        self._runtime_options = runtime_options
        self._evaluation_options = evaluation_options

    def create[TBatch, *TModelInput, TModelOutput, TLossInput](
        self,
        model: TypedModule[*TModelInput, TModelOutput],
        collator: ModelTrainingCollator[TBatch, *TModelInput, TLossInput],
        policy: EvaluationPolicy[TBatch, *TModelInput, TModelOutput],
        data_provider: IDataProvider[TBatch],
    ) -> Evaluator[TBatch, *TModelInput, TModelOutput, TLossInput]:
        return Evaluator(
            model,
            collator,
            policy,
            data_provider,
            self._checkpoint_loader,
            self._runtime_options,
            self._evaluation_options,
        )
