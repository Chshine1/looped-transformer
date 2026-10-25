"""Serializable state of a training run."""

from dataclasses import dataclass


@dataclass
class TrainingState:
    stage_id: str
    epoch: int
    global_step: int
