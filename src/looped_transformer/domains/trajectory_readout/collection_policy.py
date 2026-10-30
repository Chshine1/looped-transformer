from typing import cast, override

import torch
from torch import Tensor, nn
from torch.utils.hooks import RemovableHandle

from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.evaluation import EvaluationPolicy
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.text_classification.model import LoopedTextClassifier
from looped_transformer.domains.text_classification.types import ModelForwardOutput
from looped_transformer.domains.trajectory_readout.batch import TrajectoryBatch


class TrajectoryCollectionPolicy(
    EvaluationPolicy[MNLIBatch, Tensor, Tensor, Tensor, ModelForwardOutput]
):
    """Collect aligned loop states without changing the backbone interface."""

    def __init__(self, max_iterations: int) -> None:
        if max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        self._states: list[Tensor | None] = [None] * max_iterations
        self._padding_mask: Tensor | None = None
        self._collected: TrajectoryBatch | None = None
        self._hook_handles: list[RemovableHandle] = []

    @override
    def reset(
        self,
        model: TypedModule[Tensor, Tensor, Tensor, ModelForwardOutput],
    ) -> None:
        self.close()
        self._clear_capture()

        def capture_inputs(
            _module: nn.Module, input_args: tuple[Tensor, Tensor, Tensor]
        ) -> None:
            if self._collected is not None:
                raise RuntimeError("The previous trajectory has not been consumed")
            self._states[:] = [None] * len(self._states)
            self._padding_mask = input_args[1].detach()

        def capture_aligned_state(
            _module: nn.Module,
            input_args: tuple[int, Tensor],
            output: Tensor,
        ) -> None:
            rotation, _state = input_args
            index = -rotation - 1
            if index < 0 or index >= len(self._states):
                raise RuntimeError(f"Unexpected alignment rotation {rotation}")
            self._states[index] = output.detach()

        handles: list[RemovableHandle] = []
        try:
            handles.append(model.torch.register_forward_pre_hook(capture_inputs))
            backbone = cast(LoopedTextClassifier, model.torch)
            handles.append(
                backbone._alignment_rotation.register_forward_hook(
                    capture_aligned_state
                )
            )
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        self._hook_handles.extend(handles)

    @override
    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        if self._collected is not None:
            raise RuntimeError("The previous trajectory has not been consumed")
        if self._padding_mask is None or any(state is None for state in self._states):
            raise RuntimeError("The backbone did not produce a complete trajectory")

        states = torch.stack(
            [state for state in self._states if state is not None], dim=0
        )
        self._collected = TrajectoryBatch(
            mnli=batch,
            timed_hidden_states=self._copy_to_cpu(states),
            padding_mask=self._copy_to_cpu(self._padding_mask),
            final_logits=self._copy_to_cpu(output["timed_logits"][-1]),
        )
        self._states[:] = [None] * len(self._states)
        self._padding_mask = None

    def take_batch(self) -> TrajectoryBatch:
        collected, self._collected = self._collected, None
        if collected is None:
            raise RuntimeError("No trajectory batch has been collected")
        return collected

    @override
    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        self._clear_capture()

    def _clear_capture(self) -> None:
        self._states[:] = [None] * len(self._states)
        self._padding_mask = None
        self._collected = None

    @staticmethod
    def _copy_to_cpu(tensor: Tensor) -> Tensor:
        return tensor.detach().to(device="cpu", copy=True)
