from contextlib import nullcontext

import torch
from torch import autocast

from looped_transformer.core.config.base import OptionsModel
from looped_transformer.core.training.config import (
    ReproducibilityOptions,
    TrainingRuntimeOptions,
)
from looped_transformer.utils.pydantic import TorchDType
from looped_transformer.utils.pydantic import TorchDevice


class DeviceOptions(OptionsModel):
    device: TorchDevice


class PrecisionOptions(OptionsModel):
    dtype: TorchDType
    amp_enabled: bool
    amp_dtype: TorchDType

    def autocast(self, device: torch.device) -> autocast | nullcontext:
        if (
            self.amp_enabled
            and device.type == "cpu"
            and self.amp_dtype == torch.float16
        ):
            raise ValueError(
                "float16 AMP is unsupported on CPU; use bfloat16 or disable AMP"
            )
        return (
            torch.autocast(device.type, dtype=self.amp_dtype)
            if self.amp_enabled
            else nullcontext()
        )


class RuntimeOptions(OptionsModel):
    device: DeviceOptions
    precision: PrecisionOptions
    reproducibility: ReproducibilityOptions
    training: TrainingRuntimeOptions
