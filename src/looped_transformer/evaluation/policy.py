from typing import Protocol

from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.model import ModelForwardOutput


class EvaluationPolicy(Protocol):
    def reset(self) -> None:
        pass

    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        pass
