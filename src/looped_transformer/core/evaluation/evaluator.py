import torch

from looped_transformer.core.abstractions.training import (
    IDataProvider,
    ModelTrainingCollator,
)
from looped_transformer.core.abstractions.typed_module import TypedBranchedModule
from looped_transformer.core.checkpointing.loader import CheckpointLoader
from looped_transformer.core.config.base import ConfigPath, OptionsModel
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.evaluation.policy import EvaluationPolicy


class EvaluationCheckpointOptions(OptionsModel):
    path: ConfigPath


class Evaluator[TBatch, *TModelInput, TModelOutput, TLossInput]:
    def __init__(
        self,
        model: TypedBranchedModule[*TModelInput, TModelOutput],
        collator: ModelTrainingCollator[TBatch, *TModelInput, TLossInput],
        policy: EvaluationPolicy[TBatch, *TModelInput, TModelOutput],
        data_provider: IDataProvider[TBatch],
        checkpoint_loader: CheckpointLoader,
        device_options: DeviceOptions,
        precision_options: PrecisionOptions,
        checkpoint_options: EvaluationCheckpointOptions,
    ) -> None:
        self._model = model
        self._collator = collator
        self._policy = policy
        self._data_provider = data_provider

        self._device_options = device_options
        self._precision_options = precision_options
        self._training_modes: list[tuple[torch.nn.Module, bool]] | None = None

        try:
            checkpoint_loader.load(checkpoint_options.path)
            checkpoint_loader.load_model(self._model)
        finally:
            checkpoint_loader.dispose()

    def open(self) -> None:
        """Prepare the model and policy for one or more batch evaluations."""
        if self._training_modes is not None:
            raise RuntimeError("The evaluator is already open")

        training_modes = [
            (module, module.training) for module in self._model.torch.modules()
        ]
        self._training_modes = training_modes
        try:
            self._model.torch.eval()
            self._policy.reset(self._model)
        except BaseException:
            self._training_modes = None
            for module, training in training_modes:
                module.train(training)
            raise

    def evaluate_batch(self, batch: TBatch) -> list[TModelOutput]:
        """Evaluate one caller-supplied batch in the current evaluator session."""
        if self._training_modes is None:
            raise RuntimeError("The evaluator must be opened before evaluating a batch")

        micro_batches = self._collator.collate_batch(
            batch,
            self._device_options.device,
            self._precision_options.dtype,
        )
        if not micro_batches:
            raise ValueError("collate_batch() returned no micro-batches")

        outputs: list[TModelOutput] = []
        for micro_batch in micro_batches:
            with (
                torch.no_grad(),
                self._precision_options.autocast(self._device_options.device),
            ):
                output = self._model(
                    *micro_batch.inputs,
                    task=micro_batch.task,
                )
            self._policy.attend_batch(micro_batch.batch, output)
            outputs.append(output)

        return outputs

    def close(self) -> None:
        """Close the current evaluator session and restore module training modes."""
        training_modes, self._training_modes = self._training_modes, None
        if training_modes is None:
            return

        try:
            self._policy.close()
        finally:
            for module, training in training_modes:
                module.train(training)

    def evaluate(self) -> None:
        self.open()
        try:
            self._data_provider.reset_source(shuffle=False, seed=None)
            while (batch := self._data_provider.fetch()) is not None:
                self.evaluate_batch(batch)
        finally:
            try:
                self._data_provider.close()
            finally:
                self.close()
