from abc import ABC
from abc import abstractmethod

from looped_transformer.abstractions.model import TypedModule


class EvaluationPolicy[TBatch, *TModelInput, TModelOutput](ABC):
    @abstractmethod
    def reset(self, model: TypedModule[*TModelInput, TModelOutput]) -> None:
        pass

    @abstractmethod
    def attend_batch(self, batch: TBatch, output: TModelOutput) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
