from abc import ABC
from abc import abstractmethod

from looped_transformer.abstractions.data import MNLIBatch
from looped_transformer.abstractions.model import ModelForwardOutput
from looped_transformer.modules.model import LoopedMNLI


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
