from typing import TypedDict, Literal

from torch import Tensor


class ModelTrainOutput(TypedDict):
    mode: Literal["train"]
    timed_logits: Tensor
    halt_probabilities: Tensor


class ModelEvalOutput(TypedDict):
    mode: Literal["eval"]
    logits: Tensor


ModelForwardOutput = ModelTrainOutput | ModelEvalOutput
