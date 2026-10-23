from abc import ABC
from abc import abstractmethod

from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.model import ModelForwardOutput


class EvaluationPolicy(ABC):
    @abstractmethod
    def reset(self, model: LoopedMNLI) -> None:
        pass

    @abstractmethod
    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
