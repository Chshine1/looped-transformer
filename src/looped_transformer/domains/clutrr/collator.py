from typing import override

import torch
from torch import Tensor

from looped_transformer.core.abstractions.training import (
    CollatedBatch,
    ModelTrainingCollator,
)
from looped_transformer.domains.clutrr.types import CLUTRRBatch, relation_index
from looped_transformer.domains.text_classification.loss import (
    BackboneLossInput,
)
from looped_transformer.domains.text_classification.tokenizer import PairTokenizer


class CLUTRRCollator(
    ModelTrainingCollator[CLUTRRBatch, Tensor, Tensor, Tensor, BackboneLossInput]
):
    def __init__(self, tokenizer: PairTokenizer):
        self._tokenizer = tokenizer

    @override
    def collate_batch(
        self, batch: CLUTRRBatch, device: torch.device, dtype: torch.dtype
    ) -> list[CollatedBatch[CLUTRRBatch, Tensor, Tensor, Tensor, BackboneLossInput]]:
        encoding = self._tokenizer.encode(batch.stories, batch.queries)
        labels = torch.tensor(
            [relation_index(label) for label in batch.labels],
            dtype=torch.long,
            device=device,
        )
        return [
            CollatedBatch(
                batch=batch,
                task=None,
                inputs=(
                    encoding.input_ids.to(device),
                    encoding.attention_mask.to(device),
                    encoding.token_type_ids.to(device),
                ),
                targets=BackboneLossInput(labels=labels),
                samples_count=len(batch.labels),
            )
        ]
