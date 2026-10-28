import os
from typing import Annotated, Any, override

import torch
from pydantic import BeforeValidator, Field
from torch import Tensor
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from looped_transformer.core.abstractions.training import ModelTrainingCollator
from looped_transformer.core.config.base import (
    NonEmptyString,
    NonNegativeFloat,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.domains.mnli.modules.anchored_sampled_ce_loss import (
    BackboneLossInput,
)
from looped_transformer.domains.mnli.services.tokenizer import MNLITokenizer
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.mnli.types.logic_relation import LogicRelation
from looped_transformer.utils.pydantic import TorchDType


class ExternalTeacherOptions(OptionsModel):
    pretrained_model: NonEmptyString
    revision: NonEmptyString = "main"
    max_sequence_length: PositiveInt = 128
    dtype: TorchDType = torch.float16
    local_files_only: bool = False
    label_order: Annotated[
        tuple[NonEmptyString, ...],
        BeforeValidator(tuple),
        Field(min_length=3, max_length=3),
    ]


def _normalize_label(value: str) -> str:
    return value.strip().lower().replace("-", "_")


class BackboneTrainingCollator(
    ModelTrainingCollator[MNLIBatch, Tensor, Tensor, Tensor, BackboneLossInput]
):
    def __init__(
        self,
        student_tokenizer: MNLITokenizer,
        teacher_options: ExternalTeacherOptions,
        distillation_scale: NonNegativeFloat,
        device: torch.device,
    ) -> None:
        self._student_tokenizer = student_tokenizer
        self._teacher_tokenizer: Any | None = None
        self._teacher: Any | None = None
        self._teacher_to_project: tuple[int, ...] | None = None
        self._teacher_max_sequence_length = teacher_options.max_sequence_length

        if distillation_scale == 0:
            return

        offline = teacher_options.local_files_only or os.environ.get(
            "HF_HUB_OFFLINE", ""
        ).lower() in {"1", "true", "yes"}
        load_options: dict[str, Any] = {
            "revision": teacher_options.revision,
            "local_files_only": offline,
        }
        self._teacher_tokenizer = AutoTokenizer.from_pretrained(
            teacher_options.pretrained_model,
            **load_options,
        )
        self._teacher = AutoModelForSequenceClassification.from_pretrained(
            teacher_options.pretrained_model,
            **load_options,
        ).to(device=device, dtype=teacher_options.dtype)
        self._teacher.eval().requires_grad_(False)

        normalized = tuple(
            _normalize_label(label) for label in teacher_options.label_order
        )
        expected = {"entailment", "contradiction", "neutral"}
        if set(normalized) != expected:
            raise ValueError(
                "teacher label_order must contain entailment, contradiction, neutral"
            )
        configured_labels = getattr(self._teacher.config, "id2label", {})
        inferred = tuple(
            _normalize_label(str(configured_labels.get(index, "")))
            for index in range(3)
        )
        if set(inferred) == expected and inferred != normalized:
            raise ValueError(
                "configured teacher label_order does not match model config: "
                f"configured={normalized}, model={inferred}"
            )
        project_order = ("entailment", "contradiction", "neutral")
        self._teacher_to_project = tuple(
            normalized.index(label) for label in project_order
        )

    @override
    def collate_batch(
        self,
        batch: MNLIBatch,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[tuple[Tensor, Tensor, Tensor], BackboneLossInput]:
        student_encoding = self._student_tokenizer.encode(
            batch.premises, batch.hypotheses
        )
        input_ids = student_encoding.input_ids.to(device)
        padding_mask = student_encoding.attention_mask.to(device)
        token_type_ids = student_encoding.token_type_ids.to(device)
        labels = torch.tensor(
            [LogicRelation.from_string(label).value for label in batch.labels],
            dtype=torch.long,
            device=device,
        )

        if self._teacher is None:
            return (input_ids, padding_mask, token_type_ids), BackboneLossInput(
                labels=labels
            )

        teacher_tokenizer = self._teacher_tokenizer
        teacher_to_project = self._teacher_to_project
        if teacher_tokenizer is None or teacher_to_project is None:
            raise RuntimeError("external teacher was only partially initialized")

        teacher_encoding = teacher_tokenizer(
            text=list(batch.premises),
            text_pair=list(batch.hypotheses),
            padding="max_length",
            truncation=True,
            max_length=self._teacher_max_sequence_length,
            return_tensors="pt",
        )
        teacher_inputs = {
            key: value.to(device) for key, value in teacher_encoding.items()
        }
        with torch.inference_mode():
            teacher_output = self._teacher(**teacher_inputs)
        teacher_logits = teacher_output.logits
        if teacher_logits.ndim != 2 or teacher_logits.shape[1] != 3:
            raise ValueError(
                "external teacher must produce exactly three classification logits"
            )
        indices = torch.tensor(
            teacher_to_project,
            dtype=torch.long,
            device=teacher_logits.device,
        )
        teacher_logits = (
            teacher_logits.index_select(1, indices).detach().clone().float()
        )
        return (input_ids, padding_mask, token_type_ids), BackboneLossInput(
            labels=labels,
            teacher_logits=teacher_logits,
        )
