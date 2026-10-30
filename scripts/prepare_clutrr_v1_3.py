"""Normalize corrected CLUTRR v1.3 output for the active experiment.

Run the official v1.3 generator from the ``develop`` branch with AMT templates
and ``template_amt.ignore_incorrect: true`` first. This script verifies that
setting, reduces the generator output to the fields used by training, and emits
an integrity manifest consumed by :class:`CLUTRRDataProvider`.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from looped_transformer.core.artifacts import get_file_fingerprint
from looped_transformer.domains.clutrr.data_provider import CLUTRRDataProvider
from looped_transformer.domains.clutrr.types import CLUTRR_RELATIONS, relation_index

ACTIVE_SPLIT_DEPTHS = {
    "train": (2, 3),
    "valid": (2, 3),
    "test": (4, 5, 6),
}


def _generator_excludes_incorrect_templates(config_path: Path) -> bool:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        return False
    template_config = config.get("template_amt")
    if isinstance(template_config, dict):
        return template_config.get("ignore_incorrect") is True
    stack = list(config.values())
    while stack:
        value = stack.pop()
        if not isinstance(value, dict):
            continue
        nested = value.get("template_amt")
        if isinstance(nested, dict) and nested.get("ignore_incorrect") is True:
            return True
        stack.extend(value.values())
    return False


def _resolve_name(value: Any, name_map: Any) -> str:
    if isinstance(name_map, dict):
        for key in (value, str(value)):
            if key in name_map:
                name = name_map[key]
                if isinstance(name, str) and name:
                    return name
    if isinstance(value, str) and value:
        return value.strip("[]")
    raise ValueError(f"Cannot resolve query entity {value!r}")


def _normalize_example(raw: dict[str, Any], source: Path, line_number: int) -> dict:
    try:
        story = raw["text_story"]
        raw_query = raw["query"]
        label = raw["target_gender"].strip().lower()
        resolution_path = raw["resolution_path"]
        edges = raw["edges"]
    except (KeyError, AttributeError) as error:
        raise ValueError(
            f"{source}:{line_number}: missing v1.3 field {error}"
        ) from error

    identifier = str(raw.get("id") or f"{source.stem}:{line_number}")

    if not isinstance(story, str) or not story.strip():
        raise ValueError(f"{source}:{line_number}: text_story must be non-empty")
    if not isinstance(raw_query, (list, tuple)) or len(raw_query) < 2:
        raise ValueError(f"{source}:{line_number}: query must start with two entities")
    if not isinstance(resolution_path, list) or len(resolution_path) < 2:
        raise ValueError(f"{source}:{line_number}: invalid resolution_path")
    expected_path_edges = {
        (resolution_path[index], resolution_path[index + 1])
        for index in range(len(resolution_path) - 1)
    }
    observed_edges = {
        (edge[0], edge[1])
        for edge in edges
        if isinstance(edge, (list, tuple)) and len(edge) == 3
    }
    if len(edges) != len(resolution_path) - 1 or observed_edges != expected_path_edges:
        raise ValueError(
            f"{source}:{line_number}: active CLUTRR story contains non-path edges"
        )
    # The official v1.3 relation store misspells the female ``un`` target as
    # ``neice`` even though its templates correctly say ``niece``. Normalize
    # that generator-level typo before validating the project vocabulary.
    label = {"neice": "niece"}.get(label, label)
    relation_index(label)

    source_name = _resolve_name(raw_query[0], raw.get("name_map"))
    sink_name = _resolve_name(raw_query[1], raw.get("name_map"))
    query = f"[{sink_name}] is the [MASK] of [{source_name}]."
    return {
        "id": identifier,
        "story": story,
        "query": query,
        "label": label,
        "depth": len(resolution_path) - 1,
    }


def _normalize_file(source: Path, destination: Path) -> int:
    rows = 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
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
                output_stream.write(json.dumps(normalized, ensure_ascii=False))
                output_stream.write("\n")
                rows += 1
        if rows == 0:
            raise ValueError(f"Generated CLUTRR split is empty: {source}")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return rows


def _audit_descriptors(input_dir: Path) -> dict[str, Any]:
    descriptors: dict[str, set[str]] = {}
    logical_ids: dict[str, set[str]] = {}
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
    descriptor_overlap = descriptors["train"] & descriptors["test"]
    if descriptor_overlap:
        raise ValueError("Train and test descriptors are not length-disjoint")

    return {
        "iid_validation_from_training_pool": True,
        "length_disjoint_train_test": True,
        "depth_counts": depth_counts,
        "unique_descriptor_counts": {
            split: len(values) for split, values in descriptors.items()
        },
        "descriptor_set_sha256": {
            split: hashlib.sha256("\n".join(sorted(values)).encode("utf-8")).hexdigest()
            for split, values in descriptors.items()
        },
    }


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
            "Run scripts/balance_clutrr_v1_3_graphs.py before rendering, then "
            "pass --balanced-graph-realizations"
        )

    if not _generator_excludes_incorrect_templates(args.generator_config):
        raise ValueError(
            "Generator config must set template_amt.ignore_incorrect: true"
        )

    split_audit = _audit_descriptors(args.input_dir)

    files: dict[str, dict[str, Any]] = {}
    for split in ("train", "valid", "test"):
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
            "label_normalizations": {"neice": "niece"},
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
