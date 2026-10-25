import random

import torch

from looped_transformer.core.config.base import (
    NonNegativeInt,
    OptionsModel,
    PositiveInt,
)


class ReproducibilityOptions(OptionsModel):
    seed: NonNegativeInt
    allow_tf32: bool
    deterministic: bool

    def activate(self, device: torch.device) -> None:
        random.seed(self.seed)
        torch.manual_seed(self.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.use_deterministic_algorithms(self.deterministic)


class TrainingRuntimeOptions(OptionsModel):
    log_every_steps: PositiveInt
