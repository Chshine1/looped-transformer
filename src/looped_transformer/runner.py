import dataclasses
import math
import random
import re
from collections.abc import Callable, Sequence
from contextlib import nullcontext

import torch

from looped_transformer.checkpoint_manager import (
    CheckpointComponents,
    CheckpointManager,
    CheckpointManagerFactory,
)
from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.runtime import RuntimeOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.data import DataSplit
from looped_transformer.types.mnli import LogicRelation, get_logic_relation
from looped_transformer.types.model import ModelEvalOutput, ModelTrainOutput
from looped_transformer.types.train import StageTemplate, TrainingState
from looped_transformer.utils.torch import resolve_device, resolve_dtype


class MNLIRunner:
    def __init__(
        self,
        tokenizer: MNLITokenizer,
        checkpoint_manager_factory: CheckpointManagerFactory,
        data_reader: MNLIDataReader,
        model_options: ModelOptions,
        data_options: DataOptions,
        runtime_options: RuntimeOptions,
    ):
        self._checkpoint_manager_factory = checkpoint_manager_factory
        self._data_reader = data_reader

        self._model_options = model_options
        self._data_options = data_options
        self._runtime_options = runtime_options
        self._device = resolve_device(runtime_options)
        self._parameter_dtype = resolve_dtype(runtime_options.dtype)
        self._amp_dtype = resolve_dtype(runtime_options.amp_dtype)
        self._configure_runtime()

        self._tokenizer = tokenizer
        self._model = LoopedMNLI(model_options, tokenizer.get_tokenizer_info()).to(
            device=self._device, dtype=self._parameter_dtype
        )

    def _configure_runtime(self) -> None:
        options = self._runtime_options
        random.seed(options.seed)
        torch.manual_seed(options.seed)
        if self._device.type == "cuda":
            torch.cuda.manual_seed_all(options.seed)
            torch.backends.cuda.matmul.allow_tf32 = options.allow_tf32
            torch.backends.cudnn.allow_tf32 = options.allow_tf32
        torch.use_deterministic_algorithms(options.deterministic)

    def _autocast(self):
        if (
            self._runtime_options.amp_enabled
            and self._device.type == "cpu"
            and self._amp_dtype == torch.float16
        ):
            raise ValueError(
                "float16 AMP is unsupported on CPU; use bfloat16 or disable AMP"
            )
        return (
            torch.autocast(self._device.type, dtype=self._amp_dtype)
            if self._runtime_options.amp_enabled
            else nullcontext()
        )

    def train(self, stages: Sequence[StageTemplate], resume_from: str | None) -> None:
        state = TrainingState(
            stage_id="",
            epoch=0,
            global_step=0,
            best_accuracy=-1.0,
        )

        checkpoint_manager = self._checkpoint_manager_factory.create(self._device)
        load_components: Callable[[CheckpointComponents], None] | None = None
        if resume_from is not None:
            state, load_components = checkpoint_manager.load(resume_from)

        beginning_stage_index = next(
            (i for i, s in enumerate(stages) if s.id == state.stage_id), 0
        )

        was_training = self._model.training
        try:
            self._model.train()
            for stage in stages[beginning_stage_index:]:
                components = self._on_stage_begin(stage, state)
                if load_components is not None:
                    # noinspection calling-non-callable
                    load_components(components)
                    load_components = None

                stage = dataclasses.replace(stage, loss=stage.loss.to(self._device))
                for _ in range(state.epoch, stage.epochs):
                    self._run_epoch(stage, state, components)
                    self._on_epoch_end(state, components, checkpoint_manager)

                state.epoch = 0
        finally:
            self._model.train(was_training)

    def _run_epoch(
        self,
        stage: StageTemplate,
        state: TrainingState,
        components: CheckpointComponents,
    ) -> None:
        training_options = self._model_options.training
        optimizer, scheduler, scaler = (
            components.optimizer,
            components.scheduler,
            components.scaler,
        )

        for batch_index, batch in enumerate(
            stage.data_reader.get_batches_iterator(
                DataSplit.TRAIN,
                shuffle=True,
                seed=self._runtime_options.seed + state.epoch,
            )
        ):
            encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)
            label_tensor = torch.tensor(
                [get_logic_relation(label).value for label in batch.labels],
                dtype=torch.long,
                device=self._device,
            )
            acc_window_start, acc_window_end = self._get_accumulation_window(
                batch_index, stage.data_reader
            )
            with self._autocast():
                output: ModelTrainOutput = self._model(
                    encoding.input_ids, encoding.attention_mask
                )
                loss = stage.loss(output, label_tensor)
                scaled_loss = loss / (acc_window_end - acc_window_start)
            scaler.scale(scaled_loss).backward()

            if batch_index + 1 != acc_window_end:
                continue

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                self._model.parameters(), training_options.max_gradient_norm
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            scale_after = scaler.get_scale()
            optimizer.zero_grad(set_to_none=True)

            if scale_after < scale_before:
                continue

            scheduler.step()
            state.global_step += 1
            if state.global_step % training_options.log_every_steps == 0:
                print(
                    f"epoch={state.epoch + 1} step={state.global_step} "
                    f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.3e}"
                )

    def _on_stage_begin(
        self, stage: StageTemplate, state: TrainingState
    ) -> CheckpointComponents:
        state.stage_id = stage.id
        for name, param in self._model.named_parameters():
            param.requires_grad = not bool(re.match(stage.frozen_patterns, name))

        optimizer = stage.optimizer_factory(
            filter(lambda p: p.requires_grad, self._model.parameters())
        )
        scheduler = stage.scheduler_factory(optimizer)
        scaler = torch.amp.GradScaler(
            self._device.type,
            enabled=self._runtime_options.amp_enabled
            and self._amp_dtype == torch.float16,
        )
        optimizer.zero_grad(set_to_none=True)

        components = CheckpointComponents(
            model=self._model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        return components

    def _on_epoch_end(
        self,
        state: TrainingState,
        components: CheckpointComponents,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        should_evaluate = (
            state.epoch % self._model_options.evaluation.evaluate_every_epochs == 0
        )
        should_save = state.epoch % self._model_options.training.save_every_epochs == 0

        if should_save:
            checkpoint_manager.save_epoch(state, components)

        if should_evaluate:
            accuracy = self.eval()
            print(f"epoch={state.epoch} validation_accuracy={accuracy:.4%}")

            if accuracy > state.best_accuracy:
                state.best_accuracy = accuracy
                checkpoint_manager.save_best(state, components)

    def _get_accumulation_window(
        self, batch_index: int, data_reader: MNLIDataReader
    ) -> tuple[int, int]:
        accumulation_steps = self._model_options.training.gradient_accumulation_steps
        batches_count = math.ceil(
            data_reader.count_examples(DataSplit.TRAIN) / self._data_options.batch_size
        )

        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, batches_count)
        return window_start, window_end

    def eval(self) -> float:
        correct = 0
        total = 0
        was_training = self._model.training

        try:
            self._model.eval()

            for batch in self._data_reader.get_batches_iterator(
                DataSplit.EVAL,
                shuffle=False,
            ):
                encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)

                with torch.inference_mode(), self._autocast():
                    output: ModelEvalOutput = self._model(
                        encoding.input_ids, encoding.attention_mask
                    )

                predictions = tuple(
                    LogicRelation(prediction.item())
                    for prediction in output["logits"].argmax(dim=-1)
                )
                targets = [get_logic_relation(label) for label in batch.labels]

                correct += sum(
                    prediction == target
                    for prediction, target in zip(predictions, targets, strict=True)
                )
                total += len(targets)
        finally:
            self._model.train(was_training)

        if total == 0:
            raise ValueError("Evaluation data contains no recognized labels")
        return correct / total
