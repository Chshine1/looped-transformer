import unittest
from pathlib import Path

from looped_transformer.domains.proofwriter.data_source import (
    ProofWriterAuxiliaryOptions,
    ProofWriterAuxiliarySource,
)


class ProofWriterAuxiliarySourceTests(unittest.TestCase):
    def test_ratio_is_applied_to_main_batch_and_sampling_is_deterministic(self) -> None:
        source = ProofWriterAuxiliarySource.__new__(ProofWriterAuxiliarySource)
        source.options = ProofWriterAuxiliaryOptions(
            file_path=Path("proofwriter.jsonl"),
            manifest_path=Path("manifest.json"),
            examples_per_main_example=0.25,
            loss_scale=0.5,
            seed=42,
        )
        source._examples = tuple(
            {
                "id": f"pw{index}",
                "theory": f"theory {index}",
                "query": f"query {index}",
                "label": ("false", "true", "unknown")[index % 3],
                "depth": None if index % 3 == 2 else 4 + index % 2,
            }
            for index in range(100)
        )
        main_ids = tuple(f"clutrr{index}" for index in range(96))

        first = source.sample_for_main_batch(main_ids)
        second = source.sample_for_main_batch(main_ids)

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(len(first.ids), 24)
        self.assertEqual(first.ids, second.ids)
        self.assertEqual(len(set(first.ids)), 24)

    def test_zero_ratio_disables_only_the_side_batch(self) -> None:
        source = ProofWriterAuxiliarySource.__new__(ProofWriterAuxiliarySource)
        source.options = ProofWriterAuxiliaryOptions(
            file_path=Path("proofwriter.jsonl"),
            manifest_path=Path("manifest.json"),
            examples_per_main_example=0.0,
            loss_scale=1.0,
            seed=42,
        )
        source._examples = ()

        self.assertIsNone(source.sample_for_main_batch(("main",)))


if __name__ == "__main__":
    unittest.main()
