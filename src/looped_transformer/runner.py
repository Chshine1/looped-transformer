import math
import random
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import replace
from typing import cast

import torch
from torch import Tensor
from transformers import TensorType

# noinspection protected-member
from transformers.models.bert.tokenization_bert import BertTokenizerFast
from transformers.utils import PaddingStrategy

from looped_transformer.checkpoint_manager import (
    CheckpointComponents,
    CheckpointManagerFactory,
)
from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.runtime import RuntimeOptions
from looped_transformer.config.tokenizer import TokenizerOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.loss import LoopedMNLILoss
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.types.data import DataSplit
from looped_transformer.types.mnli import LogicRelation, get_logic_relation
from looped_transformer.types.model import ModelEvalOutput, ModelTrainOutput
from looped_transformer.types.train import TrainingState
from looped_transformer.utils.torch import resolve_device, resolve_dtype


class MNLIRunner:
    def __init__(
        self,
        checkpoint_manager_factory: CheckpointManagerFactory,
        data_reader: MNLIDataReader,
        model_options: ModelOptions,
        data_options: DataOptions,
        tokenizer_options: TokenizerOptions,
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

        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            tokenizer_options.pretrained_model_name
        )
        resolved_tokenizer_options = replace(
            tokenizer_options,
            vocab_size=tokenizer.vocab_size,
            pad_token_id=cast(int, tokenizer.pad_token_id),
        )
        self._tokenizer = tokenizer
        self._model = LoopedMNLI(model_options, resolved_tokenizer_options).to(
            device=self._device, dtype=self._parameter_dtype
        )
        training = model_options.training
        self._loss = LoopedMNLILoss(
            expected_time=training.expected_halting_time,
            halt_scale=training.halting_loss_scale,
            aux_scale=training.auxiliary_loss_scale,
        ).to(self._device)

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

    def _encode(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> dict[str, Tensor]:
        # noinspection bad-argument-type
        encoding = self._tokenizer(
            text=premises,
            text_pair=hypotheses,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
            max_length=self._model_options.max_sequence_length,
            return_tensors=TensorType.PYTORCH,
        )
        return {
            "input_ids": encoding["input_ids"].to(self._device, non_blocking=True),
            "attention_mask": encoding["attention_mask"].to(
                self._device, non_blocking=True
            ),
        }

    def train(self, resume: str | None = None) -> None:
        training = self._model_options.training
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=training.learning_rate,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_epsilon,
            weight_decay=training.weight_decay,
        )
        row_count = self._data_reader.count_examples(DataSplit.TRAIN)
        batches_per_epoch = math.ceil(row_count / self._data_options.batch_size)
        steps_per_epoch = math.ceil(
            batches_per_epoch / training.gradient_accumulation_steps
        )
        total_steps = training.max_epochs * steps_per_epoch
        warmup_steps = max(1, int(training.warmup_ratio * total_steps))
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=training.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=training.minimum_learning_rate,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, [warmup, cosine], [warmup_steps]
        )
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
        checkpoint_manager = self._checkpoint_manager_factory.create(
            components, self._device
        )
        print(f"output_dir={checkpoint_manager.output_dir}")
        state = TrainingState(
            epoch=0,
            global_step=0,
            best_accuracy=-1.0,
            evaluation_completed=True,
        )

        if resume is not None:
            checkpoint_path = checkpoint_manager.resolve_resume_path(resume)
            state = checkpoint_manager.load(checkpoint_path)

            print(
                f"resumed_from={checkpoint_path} "
                f"completed_epochs={state.epoch} step={state.global_step}"
            )

            if not state.evaluation_completed:
                accuracy = self.eval()
                print(f"epoch={state.epoch} validation_accuracy={accuracy:.4%}")

                state.evaluation_completed = True
                if accuracy > state.best_accuracy:
                    state.best_accuracy = accuracy
                    checkpoint_manager.save_best(state)

                checkpoint_manager.save(checkpoint_path, state)

        for epoch in range(state.epoch, training.max_epochs):
            self._model.train()
            for batch_index, batch in enumerate(
                self._data_reader.get_batches_iterator(
                    DataSplit.TRAIN,
                    shuffle=True,
                    seed=self._runtime_options.seed + epoch,
                )
            ):
                encoding = self._encode(batch.premises, batch.hypotheses)
                label_tensor = torch.tensor(
                    [get_logic_relation(label).value for label in batch.labels],
                    dtype=torch.long,
                    device=self._device,
                )
                with self._autocast():
                    output: ModelTrainOutput = self._model(
                        encoding["input_ids"], encoding["attention_mask"]
                    )
                    loss = self._loss(output, label_tensor)
                    scaled_loss = loss / training.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

                final_batch = batch_index + 1 == batches_per_epoch
                if (
                    batch_index + 1
                ) % training.gradient_accumulation_steps != 0 and not final_batch:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self._model.parameters(), training.max_gradient_norm
                )
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                state.global_step += 1
                if state.global_step % training.log_every_steps == 0:
                    print(
                        f"epoch={epoch + 1} step={state.global_step} "
                        f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.3e}"
                    )

            should_evaluate = (
                epoch + 1
            ) % self._model_options.evaluation.evaluate_every_epochs == 0
            should_save = (epoch + 1) % training.save_every_epochs == 0

            state.epoch = epoch + 1
            state.evaluation_completed = not should_evaluate

            if should_save:
                checkpoint_manager.save_epoch(state.epoch, state)

            if should_evaluate:
                accuracy = self.eval()
                print(f"epoch={state.epoch} validation_accuracy={accuracy:.4%}")

                state.evaluation_completed = True

                if accuracy > state.best_accuracy:
                    state.best_accuracy = accuracy
                    checkpoint_manager.save_best(state)

                if should_save:
                    checkpoint_manager.save_epoch(state.epoch, state)

    def eval(self) -> float:
        correct = 0
        total = 0

        for batch in self._data_reader.get_batches_iterator(
            DataSplit.EVAL,
            shuffle=False,
        ):
            encoding = self._encode(batch.premises, batch.hypotheses)
            self._model.eval()

            with torch.inference_mode(), self._autocast():
                output: ModelEvalOutput = self._model(
                    encoding["input_ids"], encoding["attention_mask"]
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

        if total == 0:
            raise ValueError("Evaluation data contains no recognized labels")
        return correct / total
