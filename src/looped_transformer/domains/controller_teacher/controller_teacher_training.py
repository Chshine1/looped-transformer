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


def build_loop_quality_targets(
    timed_logits: Tensor,
    labels: Tensor,
    temperature: float,
    neighbor_smoothing: float,
    uniform_smoothing: float,
) -> Tensor:
    """Create a blurred distribution over correct loops.

    All correct loops share the base mass. Examples for which no loop is correct
    fall back to a soft distribution based on true-label cross-entropy.
    """
    loop_count, sample_count, classes = timed_logits.shape
    cross_entropy = functional.cross_entropy(
        timed_logits.float().reshape(-1, classes),
        labels.repeat(loop_count),
        reduction="none",
    ).reshape(loop_count, sample_count)
    fallback = torch.softmax(-cross_entropy / temperature, dim=0)
    correct = timed_logits.argmax(dim=-1).eq(labels.unsqueeze(0)).float()
    correct_count = correct.sum(dim=0, keepdim=True)
    correct_distribution = correct / correct_count.clamp_min(1)
    quality = torch.where(correct_count > 0, correct_distribution, fallback)

    if loop_count > 1 and neighbor_smoothing > 0:
        side_weight = neighbor_smoothing / 2
        smoothed = (1 - neighbor_smoothing) * quality
        smoothed[:-1] += side_weight * quality[1:]
        smoothed[1:] += side_weight * quality[:-1]
        # Keep mass that would otherwise fall outside the trajectory at its edge.
        smoothed[0] += side_weight * quality[0]
        smoothed[-1] += side_weight * quality[-1]
        quality = smoothed

    return (1 - uniform_smoothing) * quality + uniform_smoothing / loop_count


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
    """Predict per-loop correctness with deliberately smoothed binary targets."""

    def __init__(self, target_smoothing: float = 0.1) -> None:
        super().__init__()
        self._target_smoothing = target_smoothing

    def forward(
        self,
        loop_logits: Tensor,
        targets: ControllerTeacherLossInput,
    ) -> Tensor:
        logits = targets.timed_logits.float()
        if logits.ndim != 3:
            raise ValueError("timed_logits must have shape (loops, samples, classes)")
        if loop_logits.shape != logits.shape[:2]:
            raise ValueError(
                "loop_logits must have shape (loops, samples) matching timed_logits"
            )
        if targets.labels.shape != (logits.shape[1],):
            raise ValueError("labels must have shape (samples,)")

        correctness = logits.argmax(dim=-1).eq(targets.labels.unsqueeze(0)).float()
        smoothed_targets = correctness * (1 - 2 * self._target_smoothing) + (
            self._target_smoothing
        )
        return functional.binary_cross_entropy_with_logits(
            loop_logits.float(), smoothed_targets
        )
