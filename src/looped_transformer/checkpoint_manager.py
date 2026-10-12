import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from looped_transformer.artifacts import get_run_output_dir, write_manifest
from looped_transformer.config import (
    DataOptions,
    ModelOptions,
    RuntimeOptions,
)
from looped_transformer.types.train import TrainingState


@dataclass
class CheckpointComponents:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler


class CheckpointManager:
    BEST_NAME = "best.pt"

    def __init__(
        self,
        output_dir: Path,
        device: torch.device,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

    def save_best(self, state: TrainingState, components: CheckpointComponents) -> Path:
        path = self.output_dir / self.BEST_NAME
        return self._save(path, state, components)

    def save_epoch(
        self, state: TrainingState, components: CheckpointComponents
    ) -> Path:
        path = self.output_dir / state.stage_id / f"checkpoint-epoch-{state.epoch}.pt"
        return self._save(path, state, components)

    def _save(
        self, path: Path, state: TrainingState, components: CheckpointComponents
    ) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._construct_payload(state, components), path)
        return path

    def _construct_payload(
        self, state: TrainingState, components: CheckpointComponents
    ) -> dict:
        return {
            "stage_id": state.stage_id,
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model": components.model.state_dict(),
            "optimizer": components.optimizer.state_dict(),
            "scheduler": components.scheduler.state_dict(),
            "scaler": components.scaler.state_dict(),
            "best_accuracy": state.best_accuracy,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
        }

    def load(
        self, resume_from: str
    ) -> tuple[TrainingState, Callable[[CheckpointComponents], None]]:
        checkpoint: dict = torch.load(
            resume_from,
            map_location=self.device,
            weights_only=True,
        )

        def load_components(components: CheckpointComponents) -> None:
            components.model.load_state_dict(checkpoint["model"])
            components.optimizer.load_state_dict(checkpoint["optimizer"])
            components.scheduler.load_state_dict(checkpoint["scheduler"])
            components.scaler.load_state_dict(checkpoint["scaler"])

        self._restore_rng(checkpoint)

        epoch = int(checkpoint["epoch"])

        return TrainingState(
            stage_id=checkpoint["stage_id"],
            epoch=epoch,
            global_step=int(checkpoint["global_step"]),
            best_accuracy=float(checkpoint["best_accuracy"]),
        ), load_components

    def _restore_rng(self, checkpoint: dict) -> None:
        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])

        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())

        if self.device.type == "cuda" and checkpoint.get("cuda_random_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )


class CheckpointManagerFactory:
    def __init__(
        self,
        model_options: ModelOptions,
        data_options: DataOptions,
        runtime_options: RuntimeOptions,
    ) -> None:
        self._model_options = model_options
        self._data_options = data_options
        self._runtime_options = runtime_options

    def create(self, device: torch.device) -> CheckpointManager:
        output_dir, identifier, spec = get_run_output_dir(
            self._runtime_options,
            self._data_options,
            self._model_options,
        )
        write_manifest(output_dir, identifier, spec)
        return CheckpointManager(
            output_dir=output_dir,
            device=device,
        )
