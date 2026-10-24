from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training import ModelTrainingCollator
from looped_transformer.checkpoint_loader import CheckpointLoader
from looped_transformer.checkpoint_saver import CheckpointSaver
from looped_transformer.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.training.config import (
    ReproducibilityOptions,
    TrainingRuntimeOptions,
)
from looped_transformer.core.training.model_trainer import ModelTrainer


class ModelTrainerFactory:
    def __init__(
        self,
        checkpoint_saver: CheckpointSaver,
        checkpoint_loader: CheckpointLoader,
        device_options: DeviceOptions,
        precision_options: PrecisionOptions,
        reproducibility_options: ReproducibilityOptions,
        training_options: TrainingRuntimeOptions,
    ) -> None:
        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader
        self._device_options = device_options
        self._precision_options = precision_options
        self._reproducibility_options = reproducibility_options
        self._training_options = training_options

    def create[TBatch, *TModelInput, TModelOutput, TLossInput](
        self,
        model: TypedModule[*TModelInput, TModelOutput],
        collator: ModelTrainingCollator[TBatch, *TModelInput, TLossInput],
    ) -> ModelTrainer[TBatch, *TModelInput, TModelOutput, TLossInput]:
        return ModelTrainer(
            model,
            collator,
            self._checkpoint_saver,
            self._checkpoint_loader,
            self._device_options,
            self._precision_options,
            self._reproducibility_options,
            self._training_options,
        )
