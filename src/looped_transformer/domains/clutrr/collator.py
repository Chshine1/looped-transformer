from typing import override

import torch
from torch import Tensor

from looped_transformer.core.abstractions.training import (
    CollatedBatch,
    ModelTrainingCollator,
)
from looped_transformer.core.config.base import NonNegativeFloat, OptionsModel
from looped_transformer.domains.clutrr.types import CLUTRRBatch, relation_index
from looped_transformer.domains.proofwriter.data_source import (
    PROOFWRITER_TASK,
    ProofWriterAuxiliaryOptions,
    ProofWriterAuxiliarySource,
)
from looped_transformer.domains.proofwriter.types import proofwriter_label_index
from looped_transformer.domains.text_classification.loss import (
    BackboneLossInput,
)
from looped_transformer.domains.text_classification.tokenizer import PairTokenizer


class CLUTRRCollatorOptions(OptionsModel):
    main_loss_scale: NonNegativeFloat
    proofwriter: ProofWriterAuxiliaryOptions | None = None


class CLUTRRCollator(
    ModelTrainingCollator[CLUTRRBatch, Tensor, Tensor, Tensor, BackboneLossInput]
):
    def __init__(self, tokenizer: PairTokenizer, options: CLUTRRCollatorOptions):
        self._tokenizer = tokenizer
        self._options = options
        self._proofwriter = (
            ProofWriterAuxiliarySource(options.proofwriter)
            if options.proofwriter is not None
            else None
        )

    @override
    def collate_batch(
        self, batch: CLUTRRBatch, device: torch.device, dtype: torch.dtype
    ) -> list[CollatedBatch[CLUTRRBatch, Tensor, Tensor, Tensor, BackboneLossInput]]:
        main_encoding = self._tokenizer.encode(batch.stories, batch.queries)
        main_labels = torch.tensor(
            [relation_index(label) for label in batch.labels],
            dtype=torch.long,
            device=device,
        )
        micro_batches = [
            CollatedBatch(
                batch=batch,
                task=None,
                inputs=(
                    main_encoding.input_ids.to(device),
                    main_encoding.attention_mask.to(device),
                    main_encoding.token_type_ids.to(device),
                ),
                targets=BackboneLossInput(labels=main_labels),
                samples_count=len(batch.labels),
                loss_scale=self._options.main_loss_scale,
            )
        ]
        if self._proofwriter is None:
            return micro_batches

        side_batch = self._proofwriter.sample_for_main_batch(batch.ids)
        if side_batch is None:
            return micro_batches
        side_encoding = self._tokenizer.encode(side_batch.theories, side_batch.queries)
        side_labels = torch.tensor(
            [proofwriter_label_index(label) for label in side_batch.labels],
            dtype=torch.long,
            device=device,
        )
        micro_batches.append(
            CollatedBatch(
                batch=batch,
                task=PROOFWRITER_TASK,
                inputs=(
                    side_encoding.input_ids.to(device),
                    side_encoding.attention_mask.to(device),
                    side_encoding.token_type_ids.to(device),
                ),
                targets=BackboneLossInput(labels=side_labels),
                samples_count=len(side_batch.labels),
                loss_scale=self._proofwriter.options.loss_scale,
            )
        )
        return micro_batches
