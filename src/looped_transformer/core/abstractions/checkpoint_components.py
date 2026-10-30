from dataclasses import dataclass

import torch

from looped_transformer.core.abstractions.typed_module import TypedBranchedModule


@dataclass
class CheckpointComponents:
    model: TypedBranchedModule
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    scaler: torch.amp.GradScaler
