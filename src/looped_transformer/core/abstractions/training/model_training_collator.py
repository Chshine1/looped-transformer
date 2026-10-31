import math
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
    loss_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.samples_count <= 0:
            raise ValueError("samples_count must be positive")
        if not math.isfinite(self.loss_scale) or self.loss_scale < 0:
            raise ValueError("loss_scale must be finite and non-negative")


class ModelTrainingCollator[TBatch, *TModelInput, TLossInput](ABC):
    @abstractmethod
    def collate_batch(
        self, batch: TBatch, device: torch.device, dtype: torch.dtype
    ) -> list[CollatedBatch[TBatch, *TModelInput, TLossInput]]:
        pass
