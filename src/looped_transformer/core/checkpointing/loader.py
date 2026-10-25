import random
from pathlib import Path

import torch
from torch import nn

from looped_transformer.core.abstractions.checkpoint_components import (
    CheckpointComponents,
)
from looped_transformer.core.abstractions.training import TrainingState
from looped_transformer.core.config.runtime import DeviceOptions


class CheckpointLoader:
    def __init__(self, options: DeviceOptions):
        self.device = options.device
        self._checkpoint: dict | None = None

    def load(self, resume_from: str | Path) -> None:
        if self._checkpoint is not None:
            return
        checkpoint = torch.load(
            resume_from,
            map_location=self.device,
            weights_only=True,
        )
        self._checkpoint = checkpoint

        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])
        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())
        if self.device.type == "cuda" and checkpoint.get("cuda_random_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )

    def load_training_state(self) -> TrainingState:
        if self._checkpoint is None:
            raise RuntimeError()
        return TrainingState(
            stage_id=self._checkpoint["stage_id"],
            epoch=int(self._checkpoint["epoch"]),
            global_step=int(self._checkpoint["global_step"]),
        )

    def load_components(self, components: CheckpointComponents) -> None:
        if self._checkpoint is None:
            raise RuntimeError()
        components.model.load_state_dict(self._checkpoint["model"])
        components.optimizer.load_state_dict(self._checkpoint["optimizer"])
        components.scheduler.load_state_dict(self._checkpoint["scheduler"])
        components.scaler.load_state_dict(self._checkpoint["scaler"])

    def load_model(self, model: nn.Module) -> None:
        if self._checkpoint is None:
            raise RuntimeError()
        model.load_state_dict(self._checkpoint["model"])

    def dispose(self) -> None:
        self._checkpoint = None
