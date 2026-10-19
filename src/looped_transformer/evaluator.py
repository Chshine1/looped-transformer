from looped_transformer.types.model import ModelForwardOutput
import random
from contextlib import nullcontext

import torch

from looped_transformer.config import ModelOptions, RuntimeOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.mnli import LogicRelation
from looped_transformer.types.mnli import get_logic_relation
from looped_transformer.utils.torch import resolve_device
from looped_transformer.utils.torch import resolve_dtype


class Evaluator:
    def __init__(
        self,
        checkpoint_path: str,
        data_reader: MNLIDataReader,
        model_options: ModelOptions,
        runtime_options: RuntimeOptions,
    ):
        self._tokenizer = MNLITokenizer(model_options.max_sequence_length)
        self._model_options = model_options
        self._runtime_options = runtime_options
        self._model = LoopedMNLI(model_options, self._tokenizer.get_tokenizer_info())
        self._data_reader = data_reader
        self._device = resolve_device(runtime_options)

        checkpoint: dict = torch.load(
            checkpoint_path,
            map_location=self._device,
            weights_only=True,
        )
        self._model.load_state_dict(checkpoint["model"])

        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])

        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())

        if self._device.type == "cuda" and checkpoint.get("cuda_random_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )

    def evaluate(self) -> float:
        correct = 0
        total = 0
        training_modes = [(module, module.training) for module in self._model.modules()]

        try:
            self._model.eval()

            for batch in self._data_reader.get_batches_iterator(shuffle=False):
                encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)
                input_ids = encoding.input_ids.to(self._device)
                attention_mask = encoding.attention_mask.to(self._device)

                with torch.inference_mode(), self._autocast():
                    output: ModelForwardOutput = self._model(
                        input_ids,
                        attention_mask,
                    )

                logits = output["timed_logits"][-1]
                predictions = tuple(
                    LogicRelation(prediction.item())
                    for prediction in logits.argmax(dim=-1)
                )
                targets = [get_logic_relation(label) for label in batch.labels]

                correct += sum(
                    prediction == target
                    for prediction, target in zip(predictions, targets, strict=True)
                )
                total += len(targets)
        finally:
            for module, training in training_modes:
                module.train(training)

        if total == 0:
            raise ValueError("Evaluation data contains no recognized labels")
        return correct / total

    def _autocast(self):
        amp_dtype = resolve_dtype(self._runtime_options.amp_dtype)
        if (
            self._runtime_options.amp_enabled
            and self._device.type == "cpu"
            and amp_dtype == torch.float16
        ):
            raise ValueError(
                "float16 AMP is unsupported on CPU; use bfloat16 or disable AMP"
            )
        return (
            torch.autocast(self._device.type, dtype=amp_dtype)
            if self._runtime_options.amp_enabled
            else nullcontext()
        )
