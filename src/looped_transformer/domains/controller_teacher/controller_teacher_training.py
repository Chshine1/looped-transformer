from dataclasses import dataclass
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.core.abstractions.training import ModelTrainingCollator
from looped_transformer.domains.controller_teacher.batch import (
    ControllerTeacherBatch,
)
from looped_transformer.domains.mnli.types.logic_relation import LogicRelation


@dataclass(frozen=True, slots=True)
class ControllerTeacherLossInput:
    timed_logits: Tensor
    labels: Tensor


class ControllerTeacherCollator(
    ModelTrainingCollator[
        ControllerTeacherBatch,
        Tensor,
        Tensor,
        ControllerTeacherLossInput,
    ]
):
    """Move one collected backbone trajectory onto the teacher's device."""

    @override
    def collate_batch(
        self,
        batch: ControllerTeacherBatch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[tuple[Tensor, Tensor], ControllerTeacherLossInput]:
        hidden_states = batch.timed_hidden_states.to(device=device, dtype=dtype)
        padding_mask = batch.padding_mask.to(device=device)
        timed_logits = batch.timed_logits.to(device=device, dtype=torch.float32)
        labels = torch.tensor(
            [LogicRelation.from_string(label).value for label in batch.mnli.labels],
            dtype=torch.long,
            device=device,
        )
        return (hidden_states, padding_mask), ControllerTeacherLossInput(
            timed_logits=timed_logits,
            labels=labels,
        )


class ControllerTeacherLoss(nn.Module):
    """Expected frozen-backbone classification cost under teacher loop choices."""

    def __init__(self, incorrect_prediction_cost: float = 0.5) -> None:
        super().__init__()
        if incorrect_prediction_cost < 0:
            raise ValueError("incorrect_prediction_cost must be non-negative")
        self._incorrect_prediction_cost = incorrect_prediction_cost

    def forward(
        self,
        loop_logits: Tensor,
        targets: ControllerTeacherLossInput,
    ) -> Tensor:
        logits = targets.timed_logits.float()
        if logits.ndim != 3:
            raise ValueError(
                "timed_logits must have shape (loops, samples, classes)"
            )
        if loop_logits.shape != logits.shape[:2]:
            raise ValueError(
                "loop_logits must have shape (loops, samples) matching timed_logits"
            )

        loop_count, _, classes = logits.shape
        cross_entropy = functional.cross_entropy(
            logits.reshape(-1, classes),
            targets.labels.repeat(loop_count),
            reduction="none",
        ).reshape(loop_count, -1)
        incorrect = logits.argmax(dim=-1).ne(targets.labels).float()
        costs = cross_entropy + self._incorrect_prediction_cost * incorrect
        loop_probabilities = torch.softmax(loop_logits.float(), dim=0)
        return (loop_probabilities * costs).sum(dim=0).mean()
