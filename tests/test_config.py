import copy
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from looped_transformer.config import AppOptions, load_options
from looped_transformer.stages import ensure_builtin_stages_registered
from looped_transformer.stages.looped_mnli import LoopedMNLIStageOptions


class ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        ensure_builtin_stages_registered()
        cls.raw = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))

    def test_example_config_builds_typed_tree(self) -> None:
        options = load_options("config.yaml")

        stage = options.model.training.stage_templates["backbone_warmup"]
        self.assertIsInstance(stage, LoopedMNLIStageOptions)
        self.assertIsInstance(options.data.file_path, Path)
        self.assertEqual(stage.scheduler.type, "warmup_cosine")

    def test_unknown_fields_are_rejected_at_their_source(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["model"]["dropout"]["typo"] = 0.1

        with self.assertRaisesRegex(ValidationError, "model.dropout.typo"):
            AppOptions.model_validate(raw)

    def test_cross_field_constraints_are_enforced(self) -> None:
        raw = copy.deepcopy(self.raw)
        raw["model"]["hidden_state_dimension"] = 255

        with self.assertRaisesRegex(
            ValidationError, "divisible by attention_heads_count"
        ):
            AppOptions.model_validate(raw)

    def test_stage_subtypes_are_preserved_when_serialized(self) -> None:
        dumped = AppOptions.model_validate(self.raw).model_dump(mode="json")
        stage = dumped["model"]["training"]["stage_templates"]["backbone_warmup"]

        self.assertEqual(stage["type"], "looped_mnli")
        self.assertEqual(stage["scheduler"]["type"], "warmup_cosine")


if __name__ == "__main__":
    unittest.main()
