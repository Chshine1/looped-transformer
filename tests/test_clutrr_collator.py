import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch

from looped_transformer.domains.clutrr.collator import (
    CLUTRRCollator,
    CLUTRRCollatorOptions,
)
from looped_transformer.domains.clutrr.types import CLUTRRBatch
from looped_transformer.domains.proofwriter.data_source import PROOFWRITER_TASK
from looped_transformer.domains.proofwriter.types import ProofWriterBatch
from looped_transformer.domains.text_classification.tokenizer import PairTokenizer
from looped_transformer.domains.text_classification.tokenizer_types import (
    TokenizeResult,
)


class _Tokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    def encode(
        self, premises: tuple[str, ...], hypotheses: tuple[str, ...]
    ) -> TokenizeResult:
        self.calls.append((premises, hypotheses))
        count = len(premises)
        return TokenizeResult(
            input_ids=torch.ones(count, 5, dtype=torch.long),
            attention_mask=torch.ones(count, 5, dtype=torch.long),
            token_type_ids=torch.zeros(count, 5, dtype=torch.long),
        )


class _ProofWriterSource:
    def __init__(self) -> None:
        self.options = SimpleNamespace(loss_scale=0.3)
        self.main_ids: tuple[str, ...] | None = None

    def sample_for_main_batch(self, main_ids: tuple[str, ...]) -> ProofWriterBatch:
        self.main_ids = main_ids
        return ProofWriterBatch(
            ids=("pw1", "pw2"),
            theories=("theory 1", "theory 2"),
            queries=("query 1", "query 2"),
            labels=("unknown", "true"),
            depths=(None, 5),
        )


class CLUTRRCollatorTests(unittest.TestCase):
    def test_main_batch_owns_a_configured_proofwriter_side_microbatch(self) -> None:
        tokenizer = _Tokenizer()
        source = _ProofWriterSource()
        options = CLUTRRCollatorOptions.model_validate(
            {
                "main_loss_scale": 1.2,
                "proofwriter": {
                    "file_path": Path("proofwriter.jsonl"),
                    "manifest_path": Path("manifest.json"),
                    "examples_per_main_example": 0.25,
                    "loss_scale": 0.3,
                    "seed": 42,
                },
            }
        )
        with patch(
            "looped_transformer.domains.clutrr.collator.ProofWriterAuxiliarySource",
            return_value=source,
        ):
            collator = CLUTRRCollator(cast(PairTokenizer, tokenizer), options)
        batch = CLUTRRBatch(
            ids=("c1", "c2", "c3", "c4"),
            stories=("s1", "s2", "s3", "s4"),
            queries=("q1", "q2", "q3", "q4"),
            labels=("aunt", "brother", "daughter", "father"),
            depths=(2, 2, 3, 3),
        )

        main, side = collator.collate_batch(batch, torch.device("cpu"), torch.float32)

        self.assertIsNone(main.task)
        self.assertEqual(main.samples_count, 4)
        self.assertEqual(main.loss_scale, 1.2)
        self.assertEqual(main.targets.labels.tolist(), [0, 1, 2, 4])
        self.assertEqual(side.task, PROOFWRITER_TASK)
        self.assertEqual(side.samples_count, 2)
        self.assertEqual(side.loss_scale, 0.3)
        self.assertEqual(side.targets.labels.tolist(), [2, 1])
        self.assertIs(main.batch, batch)
        self.assertIs(side.batch, batch)
        self.assertEqual(source.main_ids, batch.ids)
        self.assertEqual(
            tokenizer.calls,
            [
                (batch.stories, batch.queries),
                (("theory 1", "theory 2"), ("query 1", "query 2")),
            ],
        )

    def test_without_side_task_still_returns_the_required_main_batch(self) -> None:
        tokenizer = _Tokenizer()
        options = CLUTRRCollatorOptions(
            main_loss_scale=1.0,
            proofwriter=None,
        )
        collator = CLUTRRCollator(cast(PairTokenizer, tokenizer), options)
        batch = CLUTRRBatch(
            ids=("c1",),
            stories=("story",),
            queries=("query",),
            labels=("uncle",),
            depths=(2,),
        )

        (main,) = collator.collate_batch(batch, torch.device("cpu"), torch.float32)

        self.assertIsNone(main.task)


if __name__ == "__main__":
    unittest.main()
