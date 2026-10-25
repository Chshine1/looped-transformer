import math
import re

import torch

from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training import (
    ModelTrainingCollator,
    TrainingStage,
    TrainingState,
)
from looped_transformer.checkpoint import CheckpointComponents
from looped_transformer.checkpoint_loader import CheckpointLoader
from looped_transformer.checkpoint_saver import CheckpointSaver
from looped_transformer.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.training.config import (
    ReproducibilityOptions,
    TrainingRuntimeOptions,
)


class ModelTrainer[TBatch, *TModelInput, TModelOutput, TLossInput]:
    def __init__(
        self,
        model: TypedModule[*TModelInput, TModelOutput],
        collator: ModelTrainingCollator[TBatch, *TModelInput, TLossInput],
        checkpoint_saver: CheckpointSaver,
        checkpoint_loader: CheckpointLoader,
        device_options: DeviceOptions,
        precision_options: PrecisionOptions,
        reproducibility_options: ReproducibilityOptions,
        training_options: TrainingRuntimeOptions,
    ):
        self._model = model
        self._collator = collator

        self._checkpoint_saver = checkpoint_saver
        self._checkpoint_loader = checkpoint_loader

        self._device_options = device_options
        self._precision_options = precision_options
        self._reproducibility_options = reproducibility_options
        self._training_options = training_options
        reproducibility_options.activate(device_options.device)

    def train(
        self,
        stage: TrainingStage[TBatch, TModelOutput, TLossInput],
        resume_from: str | None,
    ) -> None:
        state = TrainingState(
            stage_id="",
            epoch=0,
            global_step=0,
        )

        was_training = self._model.torch.training
        try:
            if resume_from is not None:
                self._checkpoint_loader.load(resume_from)
                state = self._checkpoint_loader.load_training_state()

            self._model.torch.train()
            components = self._on_stage_begin(stage, state)
            if resume_from is not None:
                self._checkpoint_loader.load_components(components)
                self._checkpoint_loader.dispose()

            stage.loss = stage.loss.to(self._device_options.device)
            for _ in range(state.epoch, stage.stage_options.epochs):
                self._run_epoch(stage, state, components)
                state.epoch += 1
                self._on_epoch_end(stage, state, components)

            state.epoch = 0
        finally:
            self._checkpoint_loader.dispose()
            self._model.torch.train(was_training)

    def _run_epoch(
        self,
        stage: TrainingStage[TBatch, TModelOutput, TLossInput],
        state: TrainingState,
        components: CheckpointComponents,
    ) -> None:
        optimizer, scheduler, scaler = (
            components.optimizer,
            components.scheduler,
            components.scaler,
        )
        stage.data_provider.reset_source(
            shuffle=True, seed=self._reproducibility_options.seed + state.epoch
        )

        try:
            batch_index = 0
            while (batch := stage.data_provider.fetch()) is not None:
                i, o = self._collator.collate_batch(
                    batch,
                    self._device_options.device,
                    self._precision_options.dtype,
                )
                acc_window_start, acc_window_end = self._get_accumulation_window(
                    stage.data_provider.get_batch_size(), batch_index, stage
                )
                with self._precision_options.autocast(self._device_options.device):
                    output = self._model(*i)
                    loss = stage.loss(output, o)
                    scaled_loss = loss / (acc_window_end - acc_window_start)
                scaler.scale(scaled_loss).backward()

                batch_index += 1
                if batch_index != acc_window_end:
                    continue

                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.torch.parameters(),
                    stage.stage_options.max_gradient_norm,
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
                if state.global_step % self._training_options.log_every_steps == 0:
                    print(
                        f"epoch={state.epoch + 1} step={state.global_step} "
                        f"lr={scheduler.get_last_lr()[0]:.3e}"
                    )
        finally:
            stage.data_provider.close()

    def _on_stage_begin(
        self, stage: TrainingStage, state: TrainingState
    ) -> CheckpointComponents:
        state.stage_id = stage.id
        for name, param in self._model.torch.named_parameters():
            param.requires_grad = not bool(
                re.match(stage.stage_options.frozen_patterns, name)
            )
        for name, module in self._model.torch.named_modules():
            if name == "":
                continue
            module.train(
                not bool(re.match(stage.stage_options.evaluation_patterns, name))
            )

        optimizer = stage.optimizer_factory(
            (
                (name, parameter)
                for name, parameter in self._model.torch.named_parameters()
                if parameter.requires_grad
            )
        )
        scheduler = stage.scheduler_factory(optimizer)
        scaler = torch.amp.GradScaler(
            self._device_options.device.type,
            enabled=self._precision_options.amp_enabled
            and self._precision_options.amp_dtype == torch.float16,
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
        stage: TrainingStage,
        state: TrainingState,
        components: CheckpointComponents,
    ) -> None:
        should_save = state.epoch % stage.stage_options.save_every_epochs == 0

        if should_save:
            self._checkpoint_saver.save_epoch(state, components)

    @staticmethod
    def _get_accumulation_window(
        batch_size: int, batch_index: int, stage: TrainingStage
    ) -> tuple[int, int]:
        accumulation_steps = stage.stage_options.gradient_accumulation_steps
        batches_count = math.ceil(stage.data_provider.get_samples_count() / batch_size)

        window_start = (batch_index // accumulation_steps) * accumulation_steps
        window_end = min(window_start + accumulation_steps, batches_count)
        return window_start, window_end
