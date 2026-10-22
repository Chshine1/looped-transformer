import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from looped_transformer.artifacts import get_run_output_dir, write_manifest
from looped_transformer.config import (
    CommonDataOptions,
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


class CheckpointSaver:
    def __init__(
        self,
        model_options: ModelOptions,
        data_options: CommonDataOptions,
        runtime_options: RuntimeOptions,
    ):
        output_dir, identifier, spec = get_run_output_dir(
            runtime_options,
            data_options,
            model_options,
        )
        write_manifest(output_dir, identifier, spec)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = runtime_options.device

        self._checkpoint: dict | None = None

    def save_epoch(
        self, state: TrainingState, components: CheckpointComponents
    ) -> Path:
        path = self.output_dir / state.stage_id / f"checkpoint-epoch-{state.epoch}.pt"
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


class CheckpointLoader:
    def __init__(
        self,
        runtime_options: RuntimeOptions,
    ):
        self.device = runtime_options.device
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
            best_accuracy=float(self._checkpoint["best_accuracy"]),
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
        if self._checkpoint is not None:
            del self._checkpoint
            self._checkpoint = None
