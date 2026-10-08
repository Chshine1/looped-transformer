import torch

from looped_transformer.config import RuntimeOptions

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def resolve_device(options: RuntimeOptions) -> torch.device:
    if options.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable; refusing to silently train on CPU."
            )
        if options.cuda_device_index >= torch.cuda.device_count():
            raise ValueError("cuda_device_index does not identify an available GPU")
        return torch.device("cuda", options.cuda_device_index)
    if options.device != "cpu":
        raise ValueError("runtime.device must be 'cuda' or 'cpu'")
    return torch.device("cpu")


def resolve_dtype(name: str) -> torch.dtype:
    try:
        return _DTYPES[name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported dtype {name!r}; choose from {tuple(_DTYPES)}"
        ) from error
