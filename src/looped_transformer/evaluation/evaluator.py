import torch
from torch import Tensor

from looped_transformer.checkpoint_manager import CheckpointLoader
from looped_transformer.config import ModelOptions, RuntimeOptions
from looped_transformer.data_reader import MNLIDataProvider
from looped_transformer.evaluation.options import EvaluationOptions
from looped_transformer.evaluation.policy import EvaluationPolicy
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.model import ModelForwardOutput


def _data_adapter(
    batch: MNLIBatch, device: torch.device, tokenizer: MNLITokenizer
) -> tuple[Tensor, Tensor]:
    encoding = tokenizer.encode(batch.premises, batch.hypotheses)
    input_ids = encoding.input_ids.to(device)
    attention_mask = encoding.attention_mask.to(device)
    return input_ids, attention_mask


class Evaluator:
    def __init__(
        self,
        evaluation_options: EvaluationOptions,
        checkpoint_loader: CheckpointLoader,
        data_provider: MNLIDataProvider,
        tokenizer: MNLITokenizer,
        model_options: ModelOptions,
        runtime_options: RuntimeOptions,
        policy: EvaluationPolicy,
    ) -> None:
        self._tokenizer = tokenizer
        self._runtime_options = runtime_options
        self._data_provider = data_provider
        self._policy = policy

        self._model = LoopedMNLI(
            model_options, self._tokenizer.get_tokenizer_info()
        ).to(device=runtime_options.device, dtype=runtime_options.dtype)
        try:
            checkpoint_loader.load(evaluation_options.checkpoint_path)
            checkpoint_loader.load_model(self._model)
        finally:
            checkpoint_loader.dispose()

    def evaluate(self) -> None:
        training_modes = [(module, module.training) for module in self._model.modules()]
        try:
            self._policy.reset(self._model)
            self._data_provider.reset_source(shuffle=False, seed=None)
            self._model.eval()
            while (batch := self._data_provider.fetch()) is not None:
                i = _data_adapter(batch, self._runtime_options.device, self._tokenizer)

                with torch.no_grad(), self._runtime_options.autocast:
                    output: ModelForwardOutput = self._model(*i)

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
