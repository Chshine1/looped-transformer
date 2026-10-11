import argparse
import math
from typing import Iterable

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from looped_transformer.checkpoint_manager import (
    CheckpointManagerFactory,
)
from looped_transformer.config import load_options, ModelOptions, DataOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.loss import LoopedMNLILoss
from looped_transformer.runner import MNLIRunner
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.data import DataSplit
from looped_transformer.types.train import StageTemplate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the looped MNLI transformer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="resume from a checkpoint path, or use 'latest'",
    )
    args = parser.parse_args()
    options = load_options(args.config)
    tokenizer = MNLITokenizer(options.model)
    checkpoint_manager_factory = CheckpointManagerFactory(
        options.model,
        options.data,
        options.runtime,
    )
    data_reader = MNLIDataReader(options.runtime, options.data)
    MNLIRunner(
        tokenizer,
        checkpoint_manager_factory,
        data_reader,
        options.model,
        options.data,
        options.runtime,
    ).train(
        [_get_legacy_template(data_reader, options.model, options.data)], args.resume
    )


def _get_legacy_template(
    data_reader: MNLIDataReader, model_options: ModelOptions, data_options: DataOptions
) -> StageTemplate:
    training = model_options.training

    def get_optimizer(parameters: Iterable[nn.parameter.Parameter]) -> Optimizer:
        return torch.optim.AdamW(
            parameters,
            lr=training.learning_rate,
            betas=(training.adam_beta1, training.adam_beta2),
            eps=training.adam_epsilon,
            weight_decay=training.weight_decay,
        )

    def get_scheduler(o: Optimizer) -> LRScheduler:
        row_count = data_reader.count_examples(DataSplit.TRAIN)
        batches_per_epoch = math.ceil(row_count / data_options.batch_size)
        steps_per_epoch = math.ceil(
            batches_per_epoch / training.gradient_accumulation_steps
        )
        total_steps = training.max_epochs * steps_per_epoch
        warmup_steps = max(1, int(training.warmup_ratio * total_steps))
        warmup = torch.optim.lr_scheduler.LinearLR(
            o,
            start_factor=training.warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            o,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=training.minimum_learning_rate,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            o, [warmup, cosine], [warmup_steps]
        )

    return StageTemplate(
        name="default",
        data_reader=data_reader,
        loss=LoopedMNLILoss(
            expected_time=training.expected_halting_time,
            halt_scale=training.halting_loss_scale,
            aux_scale=training.auxiliary_loss_scale,
        ),
        frozen_patterns=r"(?!)",
        optimizer_factory=get_optimizer,
        scheduler_factory=get_scheduler,
        epochs=training.max_epochs,
    )


if __name__ == "__main__":
    main()
