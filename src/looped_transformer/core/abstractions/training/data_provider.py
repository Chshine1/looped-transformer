"""Training data source contract."""

from abc import ABC, abstractmethod


class IDataProvider[TBatch](ABC):
    @abstractmethod
    def get_samples_count(self) -> int:
        pass

    @abstractmethod
    def get_batch_size(self) -> int:
        pass

    @abstractmethod
    def reset_source(self, shuffle: bool, seed: int | None) -> None:
        pass

    @abstractmethod
    def fetch(self) -> TBatch | None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
