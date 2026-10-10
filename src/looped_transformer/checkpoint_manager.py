import random
import re
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from looped_transformer.artifacts import get_run_output_dir, write_manifest
from looped_transformer.config import (
    DataOptions,
    ModelOptions,
    RuntimeOptions,
    TokenizerOptions,
)
from looped_transformer.types.train import TrainingState


@dataclass
class CheckpointComponents:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler


class CheckpointManager:
    EPOCH_RE = re.compile(r"checkpoint-epoch-(\d+)\.pt")
    BEST_NAME = "best.pt"

    def __init__(
        self,
        output_dir: Path,
        device: torch.device,
        components: CheckpointComponents,
        evaluate_every_epochs: int,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.components = components
        self.evaluate_every_epochs = evaluate_every_epochs

    def resolve_resume_path(self, resume: str) -> Path:
        if resume != "latest":
            path = Path(resume)
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint does not exist: {path}")
            return path

        candidates: list[tuple[int, Path]] = []
        for path in self.output_dir.glob("checkpoint-epoch-*.pt"):
            match = self.EPOCH_RE.fullmatch(path.name)
            if match:
                candidates.append((int(match.group(1)), path))

        if not candidates:
            raise FileNotFoundError(f"No epoch checkpoints found in {self.output_dir}")

        return max(candidates, key=lambda item: item[0])[1]

    def epoch_path(self, epoch: int) -> Path:
        return self.output_dir / f"checkpoint-epoch-{epoch}.pt"

    def best_path(self) -> Path:
        return self.output_dir / self.BEST_NAME

    def _payload(self, state: TrainingState) -> dict:
        return {
            "epoch": state.epoch,
            "global_step": state.global_step,
            "model": self.components.model.state_dict(),
            "optimizer": self.components.optimizer.state_dict(),
            "scheduler": self.components.scheduler.state_dict(),
            "scaler": self.components.scaler.state_dict(),
            "best_accuracy": state.best_accuracy,
            "evaluation_completed": state.evaluation_completed,
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
        }

    def save(self, path: Path, state: TrainingState) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._payload(state), path)
        return path

    def save_best(self, state: TrainingState) -> Path:
        return self.save(self.best_path(), state)

    def save_epoch(self, epoch: int, state: TrainingState) -> Path:
        return self.save(self.epoch_path(epoch), state)

    def load(self, path: Path) -> TrainingState:
        checkpoint: dict = torch.load(
            path,
            map_location=self.device,
            weights_only=True,
        )

        self.components.model.load_state_dict(checkpoint["model"])
        self.components.optimizer.load_state_dict(checkpoint["optimizer"])
        self.components.scheduler.load_state_dict(checkpoint["scheduler"])

        if "scaler" in checkpoint:
            self.components.scaler.load_state_dict(checkpoint["scaler"])

        self._restore_rng(checkpoint)

        epoch = int(checkpoint["epoch"])
        evaluation_completed = bool(
            checkpoint.get(
                "evaluation_completed",
                epoch % self.evaluate_every_epochs != 0,
            )
        )

        return TrainingState(
            epoch=epoch,
            global_step=int(checkpoint.get("global_step", 0)),
            best_accuracy=float(checkpoint.get("best_accuracy", -1.0)),
            evaluation_completed=evaluation_completed,
        )

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
        tokenizer_options: TokenizerOptions,
    ) -> None:
        self._model_options = model_options
        self._data_options = data_options
        self._runtime_options = runtime_options
        self._tokenizer_options = tokenizer_options

    def create(
        self, components: CheckpointComponents, device: torch.device
    ) -> CheckpointManager:
        output_dir, identifier, spec = get_run_output_dir(
            self._runtime_options,
            self._data_options,
            self._tokenizer_options,
            self._model_options,
        )
        write_manifest(output_dir, identifier, spec)
        return CheckpointManager(
            output_dir=output_dir,
            device=device,
            components=components,
            evaluate_every_epochs=self._model_options.evaluation.evaluate_every_epochs,
        )
