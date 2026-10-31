from dataclasses import dataclass
from typing import Annotated, Literal, Self

import torch
from pydantic import BeforeValidator, Field, model_validator
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.core.abstractions.typed_module import IModule
from looped_transformer.core.config.base import (
    ExclusiveProbability,
    NonNegativeFloat,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.domains.text_classification.types import ModelForwardOutput


@dataclass(frozen=True, slots=True)
class BackboneLossInput:
    labels: Tensor


class AnchoredSampledLoopLossOptions(OptionsModel):
    type: Literal["anchored_sampled_loop_cross_entropy"]
    sampled_loops: Annotated[
        tuple[PositiveInt, ...], BeforeValidator(tuple), Field(min_length=1)
    ]
    sampled_weights: Annotated[
        tuple[NonNegativeFloat, ...], BeforeValidator(tuple), Field(min_length=1)
    ]
    sampled_loss_scale: NonNegativeFloat
    label_smoothing: ExclusiveProbability = 0.0
    time_rope_regularization_scale: NonNegativeFloat = 0.0

    @model_validator(mode="after")
    def validate_samples(self) -> Self:
        if len(self.sampled_loops) != len(self.sampled_weights):
            raise ValueError("sampled_loops and sampled_weights must have equal length")
        if len(set(self.sampled_loops)) != len(self.sampled_loops):
            raise ValueError("sampled_loops must not contain duplicates")
        if not any(self.sampled_weights):
            raise ValueError("at least one sampled-loop weight must be positive")
        return self


class AnchoredSampledLoopCrossEntropyLoss(
    nn.Module, IModule[ModelForwardOutput, BackboneLossInput, Tensor]
):
    def __init__(
        self,
        options: AnchoredSampledLoopLossOptions,
    ):
        super().__init__()
        self._sampled_loops = tuple(loop - 1 for loop in options.sampled_loops)
        weight_tensor = torch.tensor(options.sampled_weights, dtype=torch.float32)
        self.register_buffer("_sampled_weights", weight_tensor / weight_tensor.sum())
        self._sampled_loss_scale = options.sampled_loss_scale
        self._label_smoothing = options.label_smoothing
        self._time_rope_regularization_scale = options.time_rope_regularization_scale

    def forward(self, output: ModelForwardOutput, targets: BackboneLossInput) -> Tensor:
        labels = targets.labels
        cross_entropy = self._cross_entropy_by_time(
            output, labels, label_smoothing=self._label_smoothing
        )
        if max(self._sampled_loops) >= cross_entropy.size(0) - 1:
            raise ValueError(
                "sampled loops must be earlier than the final model loop "
                f"{cross_entropy.size(0)}"
            )

        sampled_candidates = torch.multinomial(
            self._sampled_weights, labels.numel(), replacement=True
        )
        loop_indices = torch.tensor(
            self._sampled_loops, device=cross_entropy.device, dtype=torch.long
        )[sampled_candidates]
        batch_indices = torch.arange(labels.numel(), device=cross_entropy.device)
        sampled_loss = cross_entropy[loop_indices, batch_indices].mean()
        task_loss = cross_entropy[-1].mean() + self._sampled_loss_scale * sampled_loss

        if self._time_rope_regularization_scale == 0:
            return task_loss
        return task_loss + (
            self._time_rope_regularization_scale * output["time_rope_regularization"]
        )

    @staticmethod
    def _cross_entropy_by_time(
        output: ModelForwardOutput, labels: Tensor, label_smoothing: float
    ) -> Tensor:
        logits = output["timed_logits"].float()
        iterations, _, classes = logits.shape
        return functional.cross_entropy(
            logits.reshape(-1, classes),
            labels.unsqueeze(0).expand(iterations, -1).reshape(-1),
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(iterations, -1)
