import random
from contextlib import nullcontext

import torch

from looped_transformer.config import ModelOptions, RuntimeOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.evaluation.policy import EvaluationPolicy
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.model import ModelForwardOutput
from looped_transformer.utils.torch import resolve_device, resolve_dtype


class Evaluator:
    def __init__(
        self,
        checkpoint_path: str,
        data_reader: MNLIDataReader,
        tokenizer: MNLITokenizer,
        model_options: ModelOptions,
        runtime_options: RuntimeOptions,
        policy: EvaluationPolicy,
    ) -> None:
        self._tokenizer = tokenizer
        self._runtime_options = runtime_options
        self._data_reader = data_reader
        self._device = resolve_device(runtime_options)
        self._policy = policy

        self._model = LoopedMNLI(
            model_options, self._tokenizer.get_tokenizer_info()
        ).to(device=self._device, dtype=resolve_dtype(runtime_options.dtype))
        self._load_checkpoint(checkpoint_path)

    def evaluate(self) -> None:
        training_modes = [(module, module.training) for module in self._model.modules()]
        self._policy.reset(self._model)

        try:
            self._model.eval()
            for batch in self._data_reader.get_batches_iterator(shuffle=False):
                encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)
                input_ids = encoding.input_ids.to(self._device)
                attention_mask = encoding.attention_mask.to(self._device)

                with torch.inference_mode(), self._autocast():
                    output: ModelForwardOutput = self._model(input_ids, attention_mask)

                self._policy.attend_batch(batch, output)
        finally:
            for module, training in training_modes:
                module.train(training)

    def _load_checkpoint(self, checkpoint_path: str) -> None:
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
