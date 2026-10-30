from torch import Tensor

from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.domains.text_classification.model import (
    LoopedTextClassifier,
    ModelOptions,
)
from looped_transformer.domains.text_classification.tokenizer import PairTokenizer
from looped_transformer.domains.text_classification.types import ModelForwardOutput


def create_model(
    model_options: ModelOptions,
    sequence_options: SequenceOptions,
    tokenizer: PairTokenizer,
    device_options: DeviceOptions,
    precision_options: PrecisionOptions,
) -> TypedModule[Tensor, Tensor, Tensor, ModelForwardOutput]:
    model = LoopedTextClassifier(
        model_options, sequence_options, tokenizer.get_tokenizer_info()
    ).to(device=device_options.device, dtype=precision_options.dtype)
    return TypedModule(model)
