from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from looped_transformer.core.artifacts import get_file_fingerprint
from looped_transformer.domains.clutrr.data_provider import CLUTRRDataProvider
from looped_transformer.domains.clutrr.types import CLUTRR_RELATIONS

SplitName = Literal["train", "valid", "test"]

SPLITS: tuple[SplitName, ...] = ("train", "valid", "test")
ACTIVE_SPLIT_DEPTHS: dict[SplitName, tuple[int, ...]] = {
    "train": (2, 3),
    "valid": (2, 3),
    "test": (4, 5, 6),
}
# noinspection SpellCheckingInspection
# The official v1.3 relation store misspells the female ``un`` target
LABEL_NORMALIZATIONS: dict[str, str] = {"neice": "niece"}


class RawV13Row(BaseModel):
    model_config = ConfigDict(extra="allow")

    text_story: str
    query: list[Any]
    target_gender: str
    resolution_path: list[Any]
    edges: list[Any]
    name_map: dict[Any, Any] | None = None
    id: str | None = None

    @model_validator(mode="after")
    def _shape_is_valid(self) -> "RawV13Row":
        if not self.text_story.strip():
            raise ValueError("text_story must be non-empty")
        if len(self.query) < 2:
            raise ValueError("query must start with two entities")
        if len(self.resolution_path) < 2:
            raise ValueError("invalid resolution_path")
        expected_path_edges = {
            (self.resolution_path[i], self.resolution_path[i + 1])
            for i in range(len(self.resolution_path) - 1)
        }
        observed_edges = {
            (edge[0], edge[1])
            for edge in self.edges
            if isinstance(edge, (list, tuple)) and len(edge) == 3
        }
        if (
            len(self.edges) != len(self.resolution_path) - 1
            or observed_edges != expected_path_edges
        ):
            raise ValueError("active CLUTRR story contains non-path edges")
        return self


class NormalizedRow(BaseModel):
    id: str
    story: str
    query: str
    label: str
    depth: int


class AuditInfo(TypedDict):
    iid_validation_from_training_pool: bool
    length_disjoint_train_test: bool
    depth_counts: dict[str, dict[str, int]]
    unique_descriptor_counts: dict[str, int]
    descriptor_set_sha256: dict[str, str]


def _generator_excludes_incorrect_templates(config_path: Path) -> bool:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return False

    def has_ignore_incorrect(node: dict[str, Any]) -> bool:
        template_config = node.get("template_amt")
        if (
            isinstance(template_config, dict)
            and template_config.get("ignore_incorrect") is True
        ):
            return True
        return any(
            has_ignore_incorrect(child)
            for child in node.values()
            if isinstance(child, dict)
        )

    return has_ignore_incorrect(config)


def _resolve_name(value: Any, name_map: Mapping[Any, Any] | None) -> str:
    if name_map is not None:
        for key in (value, str(value)):
            name = name_map.get(key)
            if isinstance(name, str) and name:
                return name
    if isinstance(value, str) and value:
        return value.strip("[]")
    raise ValueError(f"Cannot resolve query entity {value!r}")


def _normalize_example(
    raw_line: dict[str, Any], source: Path, line_number: int
) -> NormalizedRow:
    try:
        raw = RawV13Row.model_validate(raw_line)
    except ValidationError as exc:
        details = "; ".join(error["msg"] for error in exc.errors())
        raise ValueError(
            f"{source}:{line_number}: invalid v1.3 row ({details})"
        ) from exc

    label = LABEL_NORMALIZATIONS.get(raw.target_gender.strip().lower())
    label = label or raw.target_gender.strip().lower()
    if label not in CLUTRR_RELATIONS:
        raise ValueError(f"{source}:{line_number}: unknown label {label!r}")

    try:
        source_name = _resolve_name(raw.query[0], raw.name_map)
        sink_name = _resolve_name(raw.query[1], raw.name_map)
    except ValueError as exc:
        raise ValueError(f"{source}:{line_number}: {exc}") from exc

    return NormalizedRow(
        id=raw.id or f"{source.stem}:{line_number}",
        story=raw.text_story,
        query=f"[{sink_name}] is the [MASK] of [{source_name}].",
        label=label,
        depth=len(raw.resolution_path) - 1,
    )


def _normalize_file(source: Path, destination: Path) -> int:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    rows = 0
    try:
        with (
            source.open("r", encoding="utf-8") as input_stream,
            temporary.open("w", encoding="utf-8", newline="\n") as output_stream,
        ):
            for line_number, line in enumerate(input_stream, 1):
                if not line.strip():
                    continue
                raw = json.loads(line)
                normalized = _normalize_example(raw, source, line_number)
                output_stream.write(
                    json.dumps(normalized.model_dump(), ensure_ascii=False)
                )
                output_stream.write("\n")
                rows += 1
        if rows == 0:
            raise ValueError(f"Generated CLUTRR split is empty: {source}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return rows


def _audit_descriptors(input_dir: Path) -> AuditInfo:
    descriptors: dict[SplitName, set[str]] = {}
    logical_ids: dict[SplitName, set[str]] = {}
    depth_counts: dict[str, dict[str, int]] = {}

    for split, expected_depths in ACTIVE_SPLIT_DEPTHS.items():
        split_descriptors: set[str] = set()
        split_logical_ids: set[str] = set()
        counts = {str(depth): 0 for depth in expected_depths}
        with (input_dir / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                descriptor = row.get("descriptor")
                if not isinstance(descriptor, str) or not descriptor:
                    raise ValueError(
                        f"{split}.jsonl:{line_number}: missing logical descriptor"
                    )
                depth = len(descriptor.split(","))
                if depth not in expected_depths:
                    raise ValueError(
                        f"{split}.jsonl:{line_number}: depth {depth} is outside "
                        f"the active {split} depths {expected_depths}"
                    )
                split_descriptors.add(descriptor)
                logical_id = row.get("logical_instance_id")
                if not isinstance(logical_id, str) or not logical_id:
                    raise ValueError(
                        f"{split}.jsonl:{line_number}: missing logical instance id"
                    )
                split_logical_ids.add(logical_id)
                counts[str(depth)] += 1
        descriptors[split] = split_descriptors
        logical_ids[split] = split_logical_ids
        depth_counts[split] = counts

    if not logical_ids["valid"].issubset(logical_ids["train"]):
        raise ValueError("IID validation must be sampled from the training pool")
    if descriptors["train"] & descriptors["test"]:
        raise ValueError("Train and test descriptors are not length-disjoint")

    return AuditInfo(
        iid_validation_from_training_pool=True,
        length_disjoint_train_test=True,
        depth_counts=depth_counts,
        unique_descriptor_counts={
            str(split): len(values) for split, values in descriptors.items()
        },
        descriptor_set_sha256={
            str(split): hashlib.sha256(
                "\n".join(sorted(values)).encode("utf-8")
            ).hexdigest()
            for split, values in descriptors.items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare corrected CLUTRR v1.3 JSONL files"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/clutrr-v1.3"))
    parser.add_argument("--generator-config", type=Path, required=True)
    parser.add_argument(
        "--source-revision",
        required=True,
        help="Exact git commit of facebookresearch/clutrr used to generate data",
    )
    parser.add_argument(
        "--balanced-graph-realizations",
        action="store_true",
        help="Confirm the active balanced logical-graph sampling step was used",
    )
    args = parser.parse_args()

    if not args.balanced_graph_realizations:
        raise ValueError(
            "Run the CLUTRR generation package before preparation, then "
            "pass --balanced-graph-realizations"
        )
    if not _generator_excludes_incorrect_templates(args.generator_config):
        raise ValueError(
            "Generator config must set template_amt.ignore_incorrect: true"
        )

    split_audit = _audit_descriptors(args.input_dir)

    files: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        source = args.input_dir / f"{split}.jsonl"
        destination = args.output_dir / f"{split}.jsonl"
        rows = _normalize_file(source, destination)
        files[destination.name] = {
            **get_file_fingerprint(destination),
            "rows": rows,
        }

    manifest = {
        "format": CLUTRRDataProvider.FORMAT,
        "source": {
            "repository": "https://github.com/facebookresearch/clutrr",
            "version": "1.3.0",
            "revision": args.source_revision,
        },
        "generation": {
            "template_source": "amt",
            "incorrect_templates_excluded": True,
            "noise": "none",
            "config": get_file_fingerprint(args.generator_config),
            "label_normalizations": LABEL_NORMALIZATIONS,
            "logical_sampling": {
                "strategy": "iid_short_balanced_with_replacement",
                "dimensions": ["depth", "ungendered_target"],
                "seed": 42,
                "split_depths": {
                    split: list(depths) for split, depths in ACTIVE_SPLIT_DEPTHS.items()
                },
                **split_audit,
            },
        },
        "labels": list(CLUTRR_RELATIONS),
        "files": files,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared corrected CLUTRR data in {args.output_dir}")


if __name__ == "__main__":
    main()
