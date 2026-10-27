from collections.abc import Mapping
from typing import Any, override

import torch
from torch import Tensor, nn

from looped_transformer.core.config.base import (
    ExclusiveProbability,
    OptionsModel,
    PositiveFloat,
)
from looped_transformer.domains.controller_teacher.controller_teacher import (
    ControllerTeacher,
)
from looped_transformer.domains.mnli.modules.model import LoopedMNLI
from looped_transformer.domains.mnli.types.model import ModelForwardOutput


class GuidedBackboneOutput(ModelForwardOutput):
    loop_supervision_weights: Tensor


class GuidanceWeightOptions(OptionsModel):
    temperature: PositiveFloat = 0.35
    neighbor_smoothing: ExclusiveProbability = 0.2
    uniform_smoothing: ExclusiveProbability = 0.05


def build_guidance_weights(
    controller_logits: Tensor, options: GuidanceWeightOptions
) -> Tensor:
    """Turn correctness logits into moderately sharpened, nonzero loop weights."""
    loop_count = controller_logits.shape[0]
    correctness = torch.sigmoid(controller_logits.float()).clamp_min(1e-8)
    weights = torch.softmax(correctness.log() / options.temperature, dim=0)

    if loop_count > 1 and options.neighbor_smoothing > 0:
        side_weight = options.neighbor_smoothing / 2
        smoothed = (1 - options.neighbor_smoothing) * weights
        smoothed[:-1] += side_weight * weights[1:]
        smoothed[1:] += side_weight * weights[:-1]
        smoothed[0] += side_weight * weights[0]
        smoothed[-1] += side_weight * weights[-1]
        weights = smoothed

    return (1 - options.uniform_smoothing) * weights + (
        options.uniform_smoothing / loop_count
    )


class ControllerGuidedBackbone(nn.Module):
    """Train a backbone with detached weights from a frozen trajectory teacher.

    Checkpoints intentionally expose only the underlying backbone state so they
    remain directly loadable by ``LoopedMNLI`` and by the next alternating cycle.
    """

    _controller: ControllerTeacher

    def __init__(
        self,
        backbone: LoopedMNLI,
        controller: ControllerTeacher,
        guidance_options: GuidanceWeightOptions,
    ) -> None:
        super().__init__()
        self._backbone = backbone
        object.__setattr__(self, "_controller", controller)
        self._guidance_options = guidance_options
        controller.requires_grad_(False)
        controller.eval()

    @override
    def train(self, mode: bool = True):
        super().train(mode)
        self._controller.eval()
        return self

    @override
    def state_dict(self, *args: Any, **kwargs: Any):
        return self._backbone.state_dict(*args, **kwargs)

    @override
    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        return self._backbone.load_state_dict(state_dict, strict=strict, assign=assign)

    def forward(self, input_ids: Tensor, padding_mask: Tensor) -> GuidedBackboneOutput:
        aligned_states: list[Tensor] = []

        def capture_aligned(
            _module: nn.Module, _inputs: tuple[Any, ...], output: Tensor
        ) -> None:
            aligned_states.append(output.detach())

        handle = self._backbone._alignment_rotation.register_forward_hook(
            capture_aligned
        )
        try:
            output = self._backbone(input_ids, padding_mask)
        finally:
            handle.remove()

        expected = output["timed_logits"].shape[0]
        if len(aligned_states) != expected:
            raise RuntimeError(
                f"Expected {expected} aligned loop states, captured {len(aligned_states)}"
            )
        with torch.no_grad():
            controller_logits = self._controller(
                torch.stack(aligned_states), padding_mask
            )
            weights = build_guidance_weights(
                controller_logits, self._guidance_options
            ).detach()

        return {
            **output,
            "loop_supervision_weights": weights,
        }
