from typing import Literal

from looped_transformer.config.base import NonNegativeInt, OptionsModel

TorchDType = Literal["float16", "bfloat16", "float32"]


class RuntimeOptions(OptionsModel):
    device: Literal["cpu", "cuda"] = "cuda"
    dtype: TorchDType = "float32"
    amp_enabled: bool = True
    amp_dtype: TorchDType = "float16"
    seed: NonNegativeInt = 42
    allow_tf32: bool = True
    deterministic: bool = False
    cuda_device_index: NonNegativeInt = 0
