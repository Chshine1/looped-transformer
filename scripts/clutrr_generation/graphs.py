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


def postprocess_graphs(graph_dir: Path, targets: DepthTargets) -> None:
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

    selected: dict[SplitName, list[GraphRecord]] = {}
    for split in SPLITS:
        selected[split] = []
        for depth, expected in targets[split].items():
            available = pools[split][depth]
            if len(available) < expected:
                raise ValueError(
                    f"Not enough unique {split} graphs at depth {depth}: "
                    f"expected {expected}, found {len(available)} after deduplication. "
                    "Increase oversample_multiplier."
                )
            selected[split].extend(available[:expected])

    for split in SPLITS:
        destination = graph_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for record in selected[split]:
                stream.write(record.model_dump_json())
                stream.write("\n")
        print(f"post-processed {split}: {len(selected[split])} graphs")
