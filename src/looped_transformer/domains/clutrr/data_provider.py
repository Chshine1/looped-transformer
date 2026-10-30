import json
import random
from collections.abc import Iterator
from pathlib import Path
from typing import Any, override

from pydantic import model_validator

from looped_transformer.core.abstractions.training import IDataProvider
from looped_transformer.core.config.base import (
    ConfigPath,
    NonNegativeInt,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.domains.clutrr.types import (
    CLUTRR_RELATIONS,
    CLUTRRBatch,
    relation_index,
)


class CLUTRRDataOptions(OptionsModel):
    seed: NonNegativeInt
    file_path: ConfigPath
    manifest_path: ConfigPath
    batch_size: PositiveInt

    @model_validator(mode="after")
    def require_separate_data_and_manifest(self):
        if self.file_path == self.manifest_path:
            raise ValueError("file_path and manifest_path must be different")
        return self


class CLUTRRDataProvider(IDataProvider[CLUTRRBatch]):
    FORMAT = "looped-transformer-clutrr-v1.3-length-extrapolation-v1"

    def __init__(self, data_options: CLUTRRDataOptions):
        self._options = data_options
        self._examples: tuple[dict[str, Any], ...] | None = None
        self._iterator: Iterator[int] | None = None

    def _get_examples(self) -> tuple[dict[str, Any], ...]:
        if self._examples is None:
            self._examples = self._load_examples(
                self._options.file_path, self._options.manifest_path
            )
        return self._examples

    @classmethod
    def _load_examples(
        cls, data_path: Path, manifest_path: Path
    ) -> tuple[dict[str, Any], ...]:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(
                "Corrected CLUTRR data requires the manifest produced by "
                "scripts/prepare_clutrr_v1_3.py"
            ) from error

        if manifest.get("format") != cls.FORMAT:
            raise ValueError(
                f"Unsupported CLUTRR manifest format: {manifest.get('format')!r}"
            )
        source = manifest.get("source", {})
        if source.get("version") != "1.3.0":
            raise ValueError("Only corrected CLUTRR v1.3.0 data is accepted")
        if source.get("repository") != "https://github.com/facebookresearch/clutrr":
            raise ValueError("CLUTRR manifest must identify the official repository")
        if not source.get("revision"):
            raise ValueError("CLUTRR manifest must pin an exact source revision")
        generation = manifest.get("generation", {})
        if generation.get("template_source") != "amt":
            raise ValueError("The active experiment requires the v1.3 AMT templates")
        if generation.get("incorrect_templates_excluded") is not True:
            raise ValueError("CLUTRR incorrect AMT templates must be excluded")
        if generation.get("noise") != "none":
            raise ValueError("The active CLUTRR experiment requires clean stories")
        if generation.get("label_normalizations") != {"neice": "niece"}:
            raise ValueError("CLUTRR v1.3 relation-store typo was not normalized")
        logical_sampling = generation.get("logical_sampling", {})
        required_sampling = {
            "strategy": "iid_short_balanced_with_replacement",
            "dimensions": ["depth", "ungendered_target"],
            "seed": 42,
            "split_depths": {
                "train": [2, 3],
                "valid": [2, 3],
                "test": [4, 5, 6],
            },
            "iid_validation_from_training_pool": True,
            "length_disjoint_train_test": True,
        }
        if any(
            logical_sampling.get(key) != value
            for key, value in required_sampling.items()
        ):
            raise ValueError(
                "CLUTRR data must use split-local sampling with train/valid "
                "depths 2-3 and length-disjoint test depths 4-6"
            )
        if tuple(manifest.get("labels", ())) != CLUTRR_RELATIONS:
            raise ValueError(
                "CLUTRR manifest label ordering does not match the project"
            )

        expected = manifest.get("files", {}).get(data_path.name)
        if expected is None:
            raise ValueError(
                f"{data_path.name!r} is not declared in the CLUTRR manifest"
            )

        from looped_transformer.core.artifacts import get_file_fingerprint

        actual = get_file_fingerprint(data_path)
        expected_fingerprint = {
            "size": expected.get("size"),
            "sha256": expected.get("sha256"),
        }
        if actual != expected_fingerprint:
            raise ValueError(f"CLUTRR data fingerprint mismatch for {data_path}")

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
                        f"Invalid CLUTRR example at {data_path}:{line_number}: {error}"
                    ) from error
                examples.append(example)
        if not examples:
            raise ValueError(f"CLUTRR split is empty: {data_path}")
        if len(examples) != expected.get("rows"):
            raise ValueError(
                f"CLUTRR row-count mismatch for {data_path}: "
                f"expected {expected.get('rows')}, got {len(examples)}"
            )
        return tuple(examples)

    @staticmethod
    def _validate_example(example: dict[str, Any]) -> None:
        required = {"id", "story", "query", "label", "depth"}
        missing = required.difference(example)
        if missing:
            raise ValueError(f"missing fields: {sorted(missing)}")
        if not all(
            isinstance(example[key], str) and example[key]
            for key in ("id", "story", "query", "label")
        ):
            raise ValueError("id, story, query, and label must be non-empty strings")
        relation_index(example["label"])
        if (
            not isinstance(example["depth"], int)
            or isinstance(example["depth"], bool)
            or example["depth"] <= 0
        ):
            raise ValueError("depth must be a positive integer")

    @override
    def reset_source(self, shuffle: bool, seed: int | None) -> None:
        indices = list(range(len(self._get_examples())))
        if shuffle:
            if seed is None:
                raise ValueError("A seed must be provided when shuffling")
            random.Random(seed).shuffle(indices)
        self._iterator = iter(indices)

    @override
    def fetch(self) -> CLUTRRBatch | None:
        if self._iterator is None:
            raise ValueError("A data source must be set before fetching batches")
        examples_source = self._get_examples()
        indices = []
        for _ in range(self._options.batch_size):
            index = next(self._iterator, None)
            if index is None:
                break
            indices.append(index)
        if not indices:
            self.close()
            return None
        examples = [examples_source[index] for index in indices]
        return CLUTRRBatch(
            ids=tuple(example["id"] for example in examples),
            stories=tuple(example["story"] for example in examples),
            queries=tuple(example["query"] for example in examples),
            labels=tuple(example["label"] for example in examples),
            depths=tuple(example["depth"] for example in examples),
        )

    @override
    def get_samples_count(self) -> int:
        return len(self._get_examples())

    @override
    def get_batch_size(self) -> int:
        return self._options.batch_size

    @override
    def close(self) -> None:
        self._iterator = None
