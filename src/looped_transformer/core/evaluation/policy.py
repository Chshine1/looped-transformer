from abc import ABC, abstractmethod

from looped_transformer.core.abstractions.typed_module import TypedModel


class EvaluationPolicy[TBatch, *TModelInput, TModelOutput](ABC):
    @abstractmethod
    def reset(self, model: TypedModel[*TModelInput, TModelOutput]) -> None:
        pass

    @abstractmethod
    def attend_batch(self, batch: TBatch, output: TModelOutput) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
