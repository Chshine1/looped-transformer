from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training import ModelTrainingCollator
from looped_transformer.checkpoint_manager import CheckpointLoader
from looped_transformer.checkpoint_manager import CheckpointSaver
from looped_transformer.config import RuntimeOptions
from looped_transformer.core.training.model_trainer import ModelTrainer


class ModelTrainerFactory:
    def __init__(
        self,
        checkpoint_saver: CheckpointSaver,
        checkpoint_loader: CheckpointLoader,
        runtime_options: RuntimeOptions,
    ) -> None:
        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader
        self._runtime_options = runtime_options

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
            self._runtime_options,
        )
