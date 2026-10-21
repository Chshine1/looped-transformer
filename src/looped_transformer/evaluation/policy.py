from looped_transformer.modules.model import LoopedMNLI
from typing import Protocol

from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.model import ModelForwardOutput


class EvaluationPolicy(Protocol):
    def reset(self, model: LoopedMNLI) -> None:
        pass

    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        pass
