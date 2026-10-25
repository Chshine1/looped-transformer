from looped_transformer.core.abstractions.model import TypedModule
from looped_transformer.core.abstractions.training import (
    IDataProvider,
    ModelTrainingCollator,
)
from looped_transformer.core.checkpointing.loader import CheckpointLoader
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.evaluation import Evaluator
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
from looped_transformer.core.evaluation.policy import EvaluationPolicy


class EvaluatorFactory:
    def __init__(
        self,
        checkpoint_loader: CheckpointLoader,
        device_options: DeviceOptions,
        precision_options: PrecisionOptions,
        checkpoint_options: EvaluationCheckpointOptions,
    ) -> None:
        self._checkpoint_loader = checkpoint_loader
        self._device_options = device_options
        self._precision_options = precision_options
        self._checkpoint_options = checkpoint_options

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
            self._device_options,
            self._precision_options,
            self._checkpoint_options,
        )
