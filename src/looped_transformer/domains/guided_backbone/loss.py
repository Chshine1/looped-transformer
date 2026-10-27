from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.core.abstractions.typed_module import IModule
from looped_transformer.core.config.base import (
    ExclusiveProbability,
    NonNegativeFloat,
    OptionsModel,
)
from looped_transformer.domains.guided_backbone.model import GuidedBackboneOutput


class ControllerWeightedLoopLossOptions(OptionsModel):
    type: Literal["controller_weighted_loop_cross_entropy"]
    label_smoothing: ExclusiveProbability = 0.05
    time_rope_regularization_scale: NonNegativeFloat = 0.01


class ControllerWeightedLoopCrossEntropyLoss(
    nn.Module, IModule[GuidedBackboneOutput, Tensor, Tensor]
):
    def __init__(self, options: ControllerWeightedLoopLossOptions) -> None:
        super().__init__()
        self._label_smoothing = options.label_smoothing
        self._time_rope_regularization_scale = options.time_rope_regularization_scale

    def forward(self, output: GuidedBackboneOutput, labels: Tensor) -> Tensor:
        logits = output["timed_logits"].float()
        weights = output["loop_supervision_weights"].float().detach()
        loops, samples, classes = logits.shape
        if weights.shape != (loops, samples):
            raise ValueError(
                "loop_supervision_weights must match timed_logits loops and samples"
            )
        if not torch.allclose(
            weights.sum(dim=0),
            torch.ones(samples, device=weights.device),
            atol=1e-5,
            rtol=1e-5,
        ):
            raise ValueError("loop_supervision_weights must sum to one per sample")

        cross_entropy = functional.cross_entropy(
            logits.reshape(-1, classes),
            labels.repeat(loops),
            reduction="none",
            label_smoothing=self._label_smoothing,
        ).reshape(loops, samples)
        task_loss = (weights * cross_entropy).sum(dim=0).mean()
        if self._time_rope_regularization_scale == 0:
            return task_loss
        return (
            task_loss
            + self._time_rope_regularization_scale * output["time_rope_regularization"]
        )
