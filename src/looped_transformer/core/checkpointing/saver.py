import random
from pathlib import Path

import torch

from looped_transformer.core.abstractions.checkpoint_components import (
    CheckpointComponents,
)
from looped_transformer.core.abstractions.training import TrainingState
from looped_transformer.core.artifacts import (
    ArtifactSpec,
    get_run_output_dir,
    write_manifest,
)
from looped_transformer.core.config.base import ConfigPath, OptionsModel
from looped_transformer.core.config.runtime import DeviceOptions


class CheckpointSavingOptions(OptionsModel):
    output_dir: ConfigPath


class CheckpointSaver:
    def __init__(
        self,
        options: CheckpointSavingOptions,
        artifact_spec: ArtifactSpec,
        device_options: DeviceOptions,
    ):
        output_dir, identifier, spec = get_run_output_dir(
            options.output_dir, artifact_spec
        )
        write_manifest(output_dir, identifier, spec)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device_options.device

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
            "python_random_state": random.getstate(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": (
                torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None
            ),
        }
