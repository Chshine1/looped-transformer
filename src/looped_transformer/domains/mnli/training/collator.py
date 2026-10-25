from typing import override

import torch
from torch import Tensor

from looped_transformer.core.abstractions.training import ModelTrainingCollator
from looped_transformer.domains.mnli.services.tokenizer import MNLITokenizer
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.mnli.types.logic_relation import LogicRelation


class MNLICollator(ModelTrainingCollator[MNLIBatch, Tensor, Tensor, Tensor]):
    def __init__(self, tokenizer: MNLITokenizer):
        self._tokenizer = tokenizer

    @override
    def collate_batch(
        self, batch: MNLIBatch, device: torch.device, dtype: torch.dtype
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        encoding = self._tokenizer.encode(batch.premises, batch.hypotheses)
        input_ids = encoding.input_ids.to(device)
        attention_mask = encoding.attention_mask.to(device)
        label_tensor = torch.tensor(
            [LogicRelation.from_string(label).value for label in batch.labels],
            dtype=torch.long,
            device=device,
        )
        return (input_ids, attention_mask), label_tensor
