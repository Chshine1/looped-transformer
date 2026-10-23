import torch
from abc import abstractmethod
from abc import ABC


class ModelTrainingCollator[TBatch, *TModelInput, TLossInput](ABC):
    @abstractmethod
    def collate_batch(
        self, batch: TBatch, device: torch.device, dtype: torch.dtype
    ) -> tuple[tuple[*TModelInput], TLossInput]:
        pass
