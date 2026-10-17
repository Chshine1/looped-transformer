from typing import Literal, NotRequired, TypedDict

from torch import Tensor


class ModelTrainOutput(TypedDict):
    mode: Literal["train"]
    timed_logits: Tensor
    halt_probabilities: Tensor
    time_rope_regularization: NotRequired[Tensor]


class ModelEvalOutput(TypedDict):
    mode: Literal["eval"]
    logits: Tensor
    iterations: Tensor


ModelForwardOutput = ModelTrainOutput | ModelEvalOutput
