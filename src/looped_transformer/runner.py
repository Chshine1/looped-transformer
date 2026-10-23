from typing import override
from looped_transformer.collator import ModelTrainingCollator
import dataclasses
import math
import random
import re
from collections.abc import Sequence

import torch
from torch import Tensor

from looped_transformer.checkpoint_manager import (
    CheckpointComponents,
    CheckpointLoader,
    CheckpointSaver,
)
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.runtime import RuntimeOptions
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.mnli import get_logic_relation
from looped_transformer.types.model import ModelForwardOutput
from looped_transformer.types.model import TypedModule
from looped_transformer.types.train import StageTemplate, TrainingState


class MNLICollator(ModelTrainingCollator[MNLIBatch, Tensor, Tensor, Tensor]):
    def __init__(
        self,
        tokenizer: MNLITokenizer,
    ):
        self._tokenizer = tokenizer

    @override
    def collate_batch(
        self, batch: MNLIBatch, device: torch.device, dtype: torch.dtype
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)
        label_tensor = torch.tensor(
            [get_logic_relation(label).value for label in batch.labels],
            dtype=torch.long,
            device=device,
        )
        return (input_ids, attention_mask), label_tensor


def _model_factory(
    model_options: ModelOptions,
    tokenizer: MNLITokenizer,
    runtime_options: RuntimeOptions,
) -> TypedModule[Tensor, Tensor, ModelForwardOutput]:
    model = LoopedMNLI(model_options, tokenizer.get_tokenizer_info()).to(
        device=runtime_options.device, dtype=runtime_options.dtype
    )
    return TypedModule(model)


class MNLIRunner:
    def __init__(
        self,
        collator: ModelTrainingCollator[MNLIBatch, Tensor, Tensor, Tensor],
        model: TypedModule[Tensor, Tensor, ModelForwardOutput],
        checkpoint_saver: CheckpointSaver,
        checkpoint_loader: CheckpointLoader,
        runtime_options: RuntimeOptions,
    ):
        self._collator = collator
        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader

        self._runtime_options = runtime_options

        self._configure_runtime()
        self._model = model

    def _configure_runtime(self) -> None:
        options = self._runtime_options
        random.seed(options.seed)
        torch.manual_seed(options.seed)
        if self._runtime_options.device.type == "cuda":
            torch.cuda.manual_seed_all(options.seed)
            torch.backends.cuda.matmul.allow_tf32 = options.allow_tf32
            torch.backends.cudnn.allow_tf32 = options.allow_tf32
        torch.use_deterministic_algorithms(options.deterministic)

    def train(self, stages: Sequence[StageTemplate], resume_from: str | None) -> None:
        state = TrainingState(
            stage_id="",
            epoch=0,
            global_step=0,
            best_accuracy=-1.0,
        )

        was_training = self._model.torch.training
        try:
            if resume_from is not None:
                self._checkpoint_loader.load(resume_from)
                state = self._checkpoint_loader.load_training_state()

            beginning_stage_index = 0
            if state.stage_id:
                beginning_stage_index = next(
                    (i for i, stage in enumerate(stages) if stage.id == state.stage_id),
                    -1,
                )
                if beginning_stage_index < 0:
                    raise ValueError(
                        f"Checkpoint stage is not configured: {state.stage_id!r}"
                    )

            self._model.torch.train()
            for stage in stages[beginning_stage_index:]:
                components = self._on_stage_begin(stage, state)
                if resume_from is not None:
                    self._checkpoint_loader.load_components(components)
                    self._checkpoint_loader.dispose()
                    resume_from = None

                stage = dataclasses.replace(
                    stage, loss=stage.loss.to(self._runtime_options.device)
                )
                for _ in range(state.epoch, stage.epochs):
                    self._run_epoch(stage, state, components)
                    state.epoch += 1
                    self._on_epoch_end(stage, state, components)

                state.epoch = 0
        finally:
            self._checkpoint_loader.dispose()
            self._model.torch.train(was_training)

    def _run_epoch(
        self,
        stage: StageTemplate,
        state: TrainingState,
        components: CheckpointComponents,
    ) -> None:
        optimizer, scheduler, scaler = (
            components.optimizer,
            components.scheduler,
            components.scaler,
        )
        stage.data_provider.reset_source(
            shuffle=True, seed=self._runtime_options.seed + state.epoch
        )

        try:
            batch_index = 0
            while (batch := stage.data_provider.fetch()) is not None:
                i, o = self._collator.collate_batch(
                    batch, self._runtime_options.device, self._runtime_options.dtype
                )
                acc_window_start, acc_window_end = self._get_accumulation_window(
                    stage.data_provider.get_batch_size(), batch_index, stage
                )
                with self._runtime_options.autocast:
                    output = self._model(*i)
                    loss = stage.loss(output, o)
                    scaled_loss = loss / (acc_window_end - acc_window_start)
                scaler.scale(scaled_loss).backward()

                if batch_index + 1 != acc_window_end:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.torch.parameters(), stage.max_gradient_norm
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
                if state.global_step % self._runtime_options.log_every_steps == 0:
                    print(
                        f"epoch={state.epoch + 1} step={state.global_step} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}"
                    )
        finally:
            stage.data_provider.close()

    def _on_stage_begin(
        self, stage: StageTemplate, state: TrainingState
    ) -> CheckpointComponents:
        state.stage_id = stage.id
        for name, param in self._model.torch.named_parameters():
            param.requires_grad = not bool(re.match(stage.frozen_patterns, name))
        for name, module in self._model.torch.named_modules():
            if name == "":
                continue
            module.train(not bool(re.match(stage.evaluation_patterns, name)))

        optimizer = stage.optimizer_factory(
            (
                (name, parameter)
                for name, parameter in self._model.torch.named_parameters()
                if parameter.requires_grad
            )
        )
        scheduler = stage.scheduler_factory(optimizer)
        scaler = torch.amp.GradScaler(
            self._runtime_options.device.type,
            enabled=self._runtime_options.amp_enabled
            and self._runtime_options.amp_dtype == torch.float16,
        )
        optimizer.zero_grad(set_to_none=True)

        components = CheckpointComponents(
            model=self._model.torch,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

        return components

    def _on_epoch_end(
        self,
        stage: StageTemplate,
        state: TrainingState,
        components: CheckpointComponents,
    ) -> None:
        should_save = state.epoch % stage.save_every_epochs == 0

        if should_save:
            self._checkpoint_saver.save_epoch(state, components)

    @staticmethod
    def _get_accumulation_window(
        batch_size: int, batch_index: int, stage: StageTemplate
    ) -> tuple[int, int]:
        accumulation_steps = stage.gradient_accumulation_steps
        batches_count = math.ceil(stage.data_provider.get_samples_count() / batch_size)

        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, batches_count)
        return window_start, window_end
