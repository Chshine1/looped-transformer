import torch

from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training import (
    ModelTrainingCollator,
    IDataProvider,
)
from looped_transformer.checkpoint_manager import CheckpointLoader
from looped_transformer.config import RuntimeOptions
from looped_transformer.config.evaluation import EvaluationOptions
from looped_transformer.core.evaluation.policy import EvaluationPolicy


class Evaluator[TBatch, *TModelInput, TModelOutput, TLossInput]:
    def __init__(
        self,
        model: TypedModule[*TModelInput, TModelOutput],
        collator: ModelTrainingCollator[TBatch, *TModelInput, TLossInput],
        policy: EvaluationPolicy[TBatch, *TModelInput, TModelOutput],
        data_provider: IDataProvider[TBatch],
        checkpoint_loader: CheckpointLoader,
        runtime_options: RuntimeOptions,
        evaluation_options: EvaluationOptions,
    ) -> None:
        self._model = model
        self._collator = collator
        self._policy = policy
        self._data_provider = data_provider

        self._runtime_options = runtime_options

        try:
            checkpoint_loader.load(evaluation_options.checkpoint_path)
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
                i, o = self._collator.collate_batch(
                    batch, self._runtime_options.device, self._runtime_options.dtype
                )

                with torch.no_grad(), self._runtime_options.autocast:
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
