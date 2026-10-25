from typing import Annotated

import torch
from pydantic import BeforeValidator


def _to_torch_device(value: object) -> torch.device:
    if isinstance(value, str):
        return torch.device(value)
    # noinspection string-conversion-without-dunder-method
    raise ValueError(f"Cannot convert {value!r} to torch.device")


TorchDevice = Annotated[torch.device, BeforeValidator(_to_torch_device)]
