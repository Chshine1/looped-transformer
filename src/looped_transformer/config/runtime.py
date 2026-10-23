from looped_transformer.config.base import PositiveInt
from contextlib import nullcontext

import torch
from torch import autocast

from looped_transformer.utils.torch import TorchDType
from looped_transformer.config.base import NonNegativeInt, OptionsModel
from looped_transformer.utils.torch import TorchDevice


class RuntimeOptions(OptionsModel):
    device: TorchDevice
    dtype: TorchDType
    amp_enabled: bool
    amp_dtype: TorchDType
    seed: NonNegativeInt
    allow_tf32: bool
    deterministic: bool
    log_every_steps: PositiveInt

    @property
    def autocast(self) -> autocast | nullcontext:
        if (
            self.amp_enabled
            and self.device.type == "cpu"
            and self.amp_dtype == torch.float16
        ):
            raise ValueError(
                "float16 AMP is unsupported on CPU; use bfloat16 or disable AMP"
            )
        return (
            torch.autocast(self.device.type, dtype=self.amp_dtype)
            if self.amp_enabled
            else nullcontext()
        )
