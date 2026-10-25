"""Checkpoint data transfer objects."""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class CheckpointComponents:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
