import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .config import SPLITS, SplitName

DEDUPLICATION_PRIORITY: tuple[SplitName, ...] = ("test", "valid", "train")
INVALID_DESCRIPTOR_FRAGMENTS = (
    "SO,SO",
    "child,inv-child",
    "inv-child,child",
    "grand-inv,grand",
    "inv-grand,grand",
    "in-law,inv-in-law",
    "inv-in-law,in-law",
    "un,inv-un",
    "inv-un,un",
)
type DepthTargets = Mapping[SplitName, Mapping[int, int]]


class GraphRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    edges: list[tuple[int, int, str]]
    source: int
    sink: int
    query: tuple[int, int]
    target: str
    descriptor: str
    rules_used: list[str]
    rules_used_pos: list[int]
    resolution_path: list[int]
    noise_edges: list[tuple[int, int, str]]

    @property
    def depth(self) -> int:
        return len(self.descriptor.split(","))

    def logical_key(self) -> str:
        supporting_edges = self.edges
        if self.noise_edges:
            noise_count = len(self.noise_edges)
            if self.edges[-noise_count:] != self.noise_edges:
                raise ValueError("noise_edges must be a suffix of edges")
            supporting_edges = self.edges[:-noise_count]
        payload = self.model_dump(exclude={"noise_edges"})
        payload["edges"] = supporting_edges
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _read_records(path: Path) -> list[GraphRecord]:
    with path.open(encoding="utf-8") as stream:
        return [
            GraphRecord.model_validate_json(line) for line in stream if line.strip()
        ]


def count_graph_records(graph_dir: Path) -> dict[SplitName, int]:
    return {split: len(_read_records(graph_dir / f"{split}.jsonl")) for split in SPLITS}


def _make_unique(records: Iterable[GraphRecord]) -> list[GraphRecord]:
    unique: list[GraphRecord] = []
    seen: set[str] = set()
    for record in records:
        key = record.logical_key()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def _is_language_compatible(record: GraphRecord) -> bool:
    return not any(
        fragment in record.descriptor for fragment in INVALID_DESCRIPTOR_FRAGMENTS
    )


def merge_depth_graphs(
    depth_directories: Mapping[int, Path], destination: Path
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        records: list[GraphRecord] = []
        for depth, directory in sorted(depth_directories.items()):
            depth_records = _read_records(directory / f"{split}.jsonl")
            records.extend(record for record in depth_records if record.depth == depth)
        with (destination / f"{split}.jsonl").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            for record in records:
                stream.write(record.model_dump_json())
                stream.write("\n")


def postprocess_graphs(graph_dir: Path, targets: DepthTargets) -> dict[SplitName, int]:
    """Write every usable candidate, leaving quota selection until rendering."""
    pools: dict[SplitName, dict[int, list[GraphRecord]]] = {}
    for split in SPLITS:
        records = _read_records(graph_dir / f"{split}.jsonl")
        pools[split] = {
            depth: _make_unique(
                record
                for record in records
                if record.depth == depth and _is_language_compatible(record)
            )
            for depth in targets[split]
        }

    all_depths = {depth for split in SPLITS for depth in targets[split]}
    for depth in all_depths:
        seen_across_splits: set[str] = set()
        for split in DEDUPLICATION_PRIORITY:
            if targets[split].get(depth, 0) == 0:
                continue
            unique_to_split: list[GraphRecord] = []
            for record in pools[split].get(depth, []):
                key = record.logical_key()
                if key not in seen_across_splits:
                    seen_across_splits.add(key)
                    unique_to_split.append(record)
            if depth in pools[split]:
                pools[split][depth] = unique_to_split

    candidates: dict[SplitName, list[GraphRecord]] = {}
    for split in SPLITS:
        candidates[split] = []
        for depth, expected in targets[split].items():
            available = pools[split][depth]
            if len(available) < expected:
                raise ValueError(
                    f"Not enough unique {split} graphs at depth {depth}: "
                    f"expected {expected}, found {len(available)} after deduplication. "
                    "Increase oversample_multiplier."
                )
            candidates[split].extend(available)

    for split in SPLITS:
        destination = graph_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for record in candidates[split]:
                stream.write(record.model_dump_json())
                stream.write("\n")
        print(f"post-processed {split}: {len(candidates[split])} candidates")
    return {split: len(candidates[split]) for split in SPLITS}


def select_rendered_graphs(graph_dir: Path, targets: DepthTargets) -> None:
    """Apply exact per-depth quotas after the language layer has succeeded."""
    selections: dict[SplitName, list[dict[str, object]]] = {}
    for split in SPLITS:
        source = graph_dir / f"{split}.jsonl"
        with source.open(encoding="utf-8") as stream:
            rendered = [json.loads(line) for line in stream if line.strip()]

        rendered_by_depth: dict[int, list[dict[str, object]]] = {}
        for depth in targets[split]:
            available = [
                row
                for row in rendered
                if isinstance(row.get("descriptor"), str)
                and len(str(row["descriptor"]).split(",")) == depth
            ]
            rendered_by_depth[depth] = available
        expected_total = sum(targets[split].values())
        depth_diagnostics = ", ".join(
            f"depth {depth}: expected {expected}, "
            f"rendered {len(rendered_by_depth[depth])}"
            for depth, expected in targets[split].items()
        )
        print(
            f"language rendering {split}: expected {expected_total}, "
            f"rendered {len(rendered)} ({depth_diagnostics})"
        )

        selected: list[dict[str, object]] = []
        for depth, expected in targets[split].items():
            available = rendered_by_depth[depth]
            if len(available) < expected:
                raise ValueError(
                    f"Not enough renderable {split} graphs at depth {depth}: "
                    f"expected {expected}, found {len(available)}. "
                    "Increase oversample_multiplier."
                )
            selected.extend(available[:expected])
        selections[split] = selected

    for split in SPLITS:
        source = graph_dir / f"{split}.jsonl"
        with source.open("w", encoding="utf-8", newline="\n") as stream:
            for row in selections[split]:
                stream.write(json.dumps(row, ensure_ascii=False))
                stream.write("\n")
        print(
            f"final {split}: expected {sum(targets[split].values())}, "
            f"wrote {len(selections[split])} samples"
        )
