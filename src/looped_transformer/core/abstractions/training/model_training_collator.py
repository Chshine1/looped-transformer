from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class CollatedBatch[TBatch, *TModelInput, TLossInput]:
    batch: TBatch
    task: str | None
    inputs: tuple[*TModelInput]
    targets: TLossInput
    samples_count: int

    def __post_init__(self) -> None:
        if self.samples_count <= 0:
            raise ValueError("samples_count must be positive")


class ModelTrainingCollator[TBatch, *TModelInput, TLossInput](ABC):
    @abstractmethod
    def collate_batch(
        self, batch: TBatch, device: torch.device, dtype: torch.dtype
    ) -> list[CollatedBatch[TBatch, *TModelInput, TLossInput]]:
        pass
