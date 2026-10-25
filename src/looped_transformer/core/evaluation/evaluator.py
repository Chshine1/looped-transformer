import torch

from looped_transformer.core.abstractions.model import TypedModule
from looped_transformer.core.abstractions.training import (
    IDataProvider,
    ModelTrainingCollator,
)
from looped_transformer.core.checkpointing.loader import CheckpointLoader
from looped_transformer.core.config.base import ConfigPath, OptionsModel
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.evaluation.policy import EvaluationPolicy


class EvaluationCheckpointOptions(OptionsModel):
    path: ConfigPath


class Evaluator[TBatch, *TModelInput, TModelOutput, TLossInput]:
    def __init__(
        self,
        model: TypedModule[*TModelInput, TModelOutput],
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

        try:
            checkpoint_loader.load(checkpoint_options.path)
            checkpoint_loader.load_model(self._model.torch)
        finally:
            checkpoint_loader.dispose()

    def evaluate(self) -> None:
        training_modes = [
            (module, module.training) for module in self._model.torch.modules()
        ]
        try:
            self._policy.reset(self._model)
            self._data_provider.reset_source(shuffle=False, seed=None)
            self._model.torch.eval()
            while (batch := self._data_provider.fetch()) is not None:
                i, _ = self._collator.collate_batch(
                    batch,
                    self._device_options.device,
                    self._precision_options.dtype,
                )

                with (
                    torch.no_grad(),
                    self._precision_options.autocast(self._device_options.device),
                ):
                    output = self._model(*i)

                self._policy.attend_batch(batch, output)
        finally:
            try:
                self._data_provider.close()
            finally:
                try:
                    self._policy.close()
                finally:
                    for module, training in training_modes:
                        module.train(training)
