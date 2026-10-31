import json
import math
import random
from pathlib import Path
from typing import Any

from pydantic import model_validator

from looped_transformer.core.artifacts import get_file_fingerprint
from looped_transformer.core.config.base import (
    ConfigPath,
    NonNegativeFloat,
    NonNegativeInt,
    OptionsModel,
)
from looped_transformer.domains.proofwriter.types import (
    PROOFWRITER_LABELS,
    ProofWriterBatch,
    proofwriter_label_index,
)

PROOFWRITER_TASK = "proofwriter"


class ProofWriterAuxiliaryOptions(OptionsModel):
    file_path: ConfigPath
    manifest_path: ConfigPath
    examples_per_main_example: NonNegativeFloat
    loss_scale: NonNegativeFloat
    seed: NonNegativeInt

    @model_validator(mode="after")
    def require_separate_data_and_manifest(self):
        if self.file_path == self.manifest_path:
            raise ValueError("file_path and manifest_path must be different")
        return self


class ProofWriterAuxiliarySource:
    FORMAT = "looped-transformer-proofwriter-owa-depth-transfer-v1"

    def __init__(self, options: ProofWriterAuxiliaryOptions) -> None:
        self.options = options
        self._examples = self._load_examples(options.file_path, options.manifest_path)

    @classmethod
    def _load_examples(
        cls, data_path: Path, manifest_path: Path
    ) -> tuple[dict[str, Any], ...]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                "ProofWriter auxiliary data requires its generated manifest"
            ) from error
        if manifest.get("format") != cls.FORMAT:
            raise ValueError(
                f"Unsupported ProofWriter manifest format: {manifest.get('format')!r}"
            )
        selection = manifest.get("selection", {})
        if selection.get("labels") != list(PROOFWRITER_LABELS):
            raise ValueError(
                "ProofWriter manifest label order does not match the project"
            )
        if selection.get("proved_and_disproved_exact_depths") != [4, 5]:
            raise ValueError("ProofWriter auxiliary data must use exact depths 4 and 5")
        if selection.get("unknown_depth", "missing") is not None:
            raise ValueError(
                "Unknown ProofWriter examples must not claim a proof depth"
            )

        expected = manifest.get("files", {}).get(data_path.name)
        if expected is None:
            raise ValueError(f"{data_path.name!r} is not declared in the manifest")
        actual = get_file_fingerprint(data_path)
        if actual != {
            "size": expected.get("size"),
            "sha256": expected.get("sha256"),
        }:
            raise ValueError(f"ProofWriter data fingerprint mismatch for {data_path}")

        examples: list[dict[str, Any]] = []
        with data_path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    example = json.loads(line)
                    cls._validate_example(example)
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    raise ValueError(
                        f"Invalid ProofWriter example at {data_path}:{line_number}: "
                        f"{error}"
                    ) from error
                examples.append(example)
        if not examples:
            raise ValueError("ProofWriter auxiliary split is empty")
        if len(examples) != expected.get("rows"):
            raise ValueError(
                f"ProofWriter row-count mismatch: expected {expected.get('rows')}, "
                f"got {len(examples)}"
            )
        return tuple(examples)

    @staticmethod
    def _validate_example(example: dict[str, Any]) -> None:
        required = {"id", "theory", "query", "label", "depth", "wordpieces"}
        missing = required.difference(example)
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        if not all(
            isinstance(example[key], str) and example[key]
            for key in ("id", "theory", "query", "label")
        ):
            raise ValueError("id, theory, query, and label must be non-empty strings")
        proofwriter_label_index(example["label"])
        depth = example["depth"]
        if example["label"] == "unknown":
            if depth is not None:
                raise ValueError("unknown examples cannot have a proof depth")
        elif depth not in (4, 5):
            raise ValueError("proved and disproved examples must have depth 4 or 5")
        wordpieces = example["wordpieces"]
        if not isinstance(wordpieces, int) or isinstance(wordpieces, bool):
            raise ValueError("wordpieces must be an integer")
        if wordpieces <= 0:
            raise ValueError("wordpieces must be positive")

    def sample_for_main_batch(
        self, main_ids: tuple[str, ...]
    ) -> ProofWriterBatch | None:
        requested = math.floor(
            len(main_ids) * self.options.examples_per_main_example + 0.5
        )
        if requested == 0:
            return None
        if requested > len(self._examples):
            raise ValueError(
                "Requested ProofWriter side batch exceeds the auxiliary dataset"
            )
        # Main IDs make sampling deterministic across restarts without mutable
        # sampler state. Shuffled CLUTRR groupings still vary it between epochs.
        rng = random.Random(f"{self.options.seed}:{'|'.join(main_ids)}")
        examples = rng.sample(self._examples, requested)
        return ProofWriterBatch(
            ids=tuple(example["id"] for example in examples),
            theories=tuple(example["theory"] for example in examples),
            queries=tuple(example["query"] for example in examples),
            labels=tuple(example["label"] for example in examples),
            depths=tuple(example["depth"] for example in examples),
        )
