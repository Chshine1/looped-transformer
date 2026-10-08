import argparse
import math
import random
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from typing import cast

import pandas as pd
import torch
from pandas.io.parsers import TextFileReader
from torch import Tensor
from transformers import TensorType

# noinspection protected-member
from transformers.models.bert.tokenization_bert import BertTokenizerFast
from transformers.utils import PaddingStrategy

from looped_transformer.config import load_options
from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.runtime import RuntimeOptions
from looped_transformer.config.tokenizer import TokenizerOptions
from looped_transformer.loss import LoopedMNLILoss
from looped_transformer.model import LoopedMNLI
from looped_transformer.types.mnli import LogicRelation, get_logic_relation
from looped_transformer.types.model import ModelEvalOutput, ModelTrainOutput
from looped_transformer.utils.torch import resolve_device, resolve_dtype


class MNLIRunner:
    def __init__(
        self,
        model_options: ModelOptions,
        data_options: DataOptions,
        tokenizer_options: TokenizerOptions,
        runtime_options: RuntimeOptions,
    ):
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

    def _encode(self, premises: list[str], hypotheses: list[str]) -> dict[str, Tensor]:
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

    def _create_checkpoint(
        self,
        epoch: int,
        global_step: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
    ):
        return {
            "epoch": epoch,
            "global_step": global_step,
            "model": self._model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }

    def _read_batches(self, file_path: str, batch_size: int) -> TextFileReader:
        # noinspection argument-list
        return pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            usecols=[
                self._data_options.premise_column,
                self._data_options.hypothesis_column,
                self._data_options.label_column,
            ],
            chunksize=batch_size,
        )

    def _compute_row_count(self, file_path: str) -> int:
        label = self._data_options.label_column
        # noinspection argument-list
        chunks = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            usecols=[label],
            chunksize=100_000,
        )
        return sum(len(chunk) for chunk in chunks)

    def train(self) -> None:
        training = self._model_options.training
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=training.learning_rate,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_epsilon,
            weight_decay=training.weight_decay,
        )
        row_count = self._compute_row_count(self._data_options.train_file_path)
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
        output_dir = Path(self._data_options.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        optimizer.zero_grad(set_to_none=True)
        global_step = 0
        best_accuracy = -1.0

        for epoch in range(training.max_epochs):
            self._model.train()
            for batch_index, chunk in enumerate(
                self._read_batches(
                    self._data_options.train_file_path,
                    self._data_options.batch_size,
                )
            ):
                encoding = self._encode(
                    chunk[self._data_options.premise_column].tolist(),
                    chunk[self._data_options.hypothesis_column].tolist(),
                )
                label_tensor = torch.tensor(
                    [
                        get_logic_relation(label).value
                        for label in chunk[self._data_options.label_column]
                    ],
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
                global_step += 1
                if global_step % training.log_every_steps == 0:
                    print(
                        f"epoch={epoch + 1} step={global_step} "
                        f"loss={loss.item():.4f} lr={scheduler.get_last_lr()[0]:.3e}"
                    )

            if (epoch + 1) % training.save_every_epochs == 0:
                checkpoint = self._create_checkpoint(
                    epoch + 1, global_step, optimizer, scheduler
                )
                torch.save(checkpoint, output_dir / f"checkpoint-epoch-{epoch + 1}.pt")

            evaluation = self._model_options.evaluation
            if (epoch + 1) % evaluation.evaluate_every_epochs == 0:
                accuracy = self.evaluate_file()
                print(f"epoch={epoch + 1} validation_accuracy={accuracy:.4%}")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    torch.save(
                        self._create_checkpoint(
                            epoch + 1, global_step, optimizer, scheduler
                        ),
                        output_dir / "best.pt",
                    )

    def evaluate_file(self) -> float:
        correct = 0
        total = 0

        # noinspection argument-list
        for chunk in self._read_batches(
            self._data_options.eval_file_path,
            self._model_options.evaluation.batch_size,
        ):
            predictions = self.evaluate(
                chunk[self._data_options.premise_column].tolist(),
                chunk[self._data_options.hypothesis_column].tolist(),
            )
            targets = [
                get_logic_relation(label)
                for label in chunk[self._data_options.label_column]
            ]
            correct += sum(
                prediction == target
                for prediction, target in zip(predictions, targets, strict=True)
            )
            total += len(targets)
        if total == 0:
            raise ValueError("Evaluation data contains no recognized labels")
        return correct / total

    def evaluate(
        self, premise_texts: list[str], hypothesis_texts: list[str]
    ) -> tuple[LogicRelation, ...]:
        encoding = self._encode(premise_texts, hypothesis_texts)
        self._model.eval()
        with torch.inference_mode(), self._autocast():
            output: ModelEvalOutput = self._model(
                encoding["input_ids"], encoding["attention_mask"]
            )
        return tuple(
            LogicRelation(prediction.item())
            for prediction in output["logits"].argmax(dim=-1)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the looped MNLI transformer")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    options = load_options(args.config)
    MNLIRunner(options.model, options.data, options.tokenizer, options.runtime).train()


if __name__ == "__main__":
    main()
