from looped_transformer.core.abstractions.training import (
    ModelTrainingCollator,
    TrainingStage,
)
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.checkpointing import CheckpointLoader, CheckpointSaver
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.training.config import (
    ReproducibilityOptions,
    TrainingRuntimeOptions,
)
from looped_transformer.core.training.model_trainer import ModelTrainer
from looped_transformer.core.training.stage import TrainingOptions


class ModelTrainerFactory:
    def __init__(
        self,
        checkpoint_saver: CheckpointSaver,
        checkpoint_loader: CheckpointLoader,
        device_options: DeviceOptions,
        precision_options: PrecisionOptions,
        reproducibility_options: ReproducibilityOptions,
        training_options: TrainingRuntimeOptions,
        stage_options: TrainingOptions,
    ) -> None:
        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader
        self._device_options = device_options
        self._precision_options = precision_options
        self._reproducibility_options = reproducibility_options
        self._training_options = training_options
        self._resume_from = (
            str(stage_options.resume_from)
            if stage_options.resume_from is not None
            else None
        )

    def create[TBatch, *TModelInput, TModelOutput, TLossInput](
        self,
        model: TypedModule[*TModelInput, TModelOutput],
        collator: ModelTrainingCollator[TBatch, *TModelInput, TLossInput],
        stage: TrainingStage[TBatch, TModelOutput, TLossInput],
    ) -> ModelTrainer[TBatch, *TModelInput, TModelOutput, TLossInput]:
        return ModelTrainer(
            model,
            collator,
            stage,
            self._resume_from,
            self._checkpoint_saver,
            self._checkpoint_loader,
            self._device_options,
            self._precision_options,
            self._reproducibility_options,
            self._training_options,
        )
