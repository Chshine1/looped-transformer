from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RuntimeOptions:
    device: str = "cuda"
    dtype: str = "float32"
    amp_enabled: bool = True
    amp_dtype: str = "float16"
    seed: int = 42
    allow_tf32: bool = True
    deterministic: bool = False
    cuda_device_index: int = 0
