from typing import Annotated

import torch
from pydantic import BeforeValidator

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _to_torch_device(value: object) -> torch.device:
    if isinstance(value, str):
        return torch.device(value)
    # noinspection string-conversion-without-dunder-method
    raise ValueError(f"Cannot convert {value!r} to torch.device")


TorchDevice = Annotated[torch.device, BeforeValidator(_to_torch_device)]


def _to_dtype(name: object) -> torch.dtype:
    if isinstance(name, str):
        try:
            return _DTYPES[name]
        except KeyError as error:
            raise ValueError(
                f"Unsupported dtype {name!r}; choose from {tuple(_DTYPES)}"
            ) from error
    # noinspection string-conversion-without-dunder-method
    raise ValueError(f"Cannot convert {name!r} to torch.dtype")


TorchDType = Annotated[torch.dtype, BeforeValidator(_to_dtype)]
