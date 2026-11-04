import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from .config import SPLITS, SplitName

DEDUPLICATION_PRIORITY: tuple[SplitName, ...] = ("test", "valid", "train")
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
    noise_edges: list[tuple[int, int, str]] = Field(default_factory=list)

    @property
    def depth(self) -> int:
        return len(self.descriptor.split(","))

    def logical_key(self) -> str:
        """Return GLC's pre-noise graph identity in a stable serialized form."""
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


def _unique(records: Iterable[GraphRecord]) -> list[GraphRecord]:
    unique: list[GraphRecord] = []
    seen: set[str] = set()
    for record in records:
        key = record.logical_key()
        if key not in seen:
            seen.add(key)
            unique.append(record)
    return unique


def postprocess_graphs(graph_dir: Path, targets: DepthTargets) -> None:
    pools: dict[SplitName, dict[int, list[GraphRecord]]] = {}
    for split in SPLITS:
        records = _read_records(graph_dir / f"{split}.jsonl")
        pools[split] = {
            depth: _unique(record for record in records if record.depth == depth)
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
                    "Increase search_multiplier."
                )
            selected[split].extend(available[:expected])

    for split in SPLITS:
        destination = graph_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for record in selected[split]:
                stream.write(record.model_dump_json())
                stream.write("\n")
        print(f"post-processed {split}: {len(selected[split])} graphs")
