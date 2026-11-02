from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

SplitName = Literal["train", "valid", "test"]
SourceName = Literal["short", "test"]

BLOCKED_DESCRIPTOR_FRAGMENTS = (
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
ACTIVE_SPLITS: dict[SplitName, dict[int, int]] = {
    "train": {2: 30_000, 3: 30_000},
    "valid": {2: 3_000, 3: 3_000},
    "test": {4: 2_000, 5: 2_000, 6: 2_000},
}
SOURCE_FILES: dict[SourceName, tuple[SplitName, ...]] = {
    "short": ("train", "valid"),
    "test": ("test",),
}
SHORT_DEPTHS = frozenset({2, 3})
LONG_DEPTHS = frozenset({4, 5, 6})
LOGICAL_ID_HASHED_FIELDS = (
    "edges",
    "source",
    "sink",
    "target",
    "descriptor",
    "rules_used",
    "rules_used_pos",
    "resolution_path",
)


class ClutrrRow(BaseModel):
    model_config = ConfigDict(extra="allow")

    descriptor: str
    target: str
    logical_instance_id: str | None = None

    @property
    def depth(self) -> int:
        return len(self.descriptor.split(","))


RowsByTarget = dict[str, list[ClutrrRow]]
PoolsByDepth = dict[int, RowsByTarget]
SplitPools = dict[SplitName, PoolsByDepth]


def _is_blocked(descriptor: str) -> bool:
    return any(fragment in descriptor for fragment in BLOCKED_DESCRIPTOR_FRAGMENTS)


def _ensure_logical_id(row: ClutrrRow) -> str:
    if row.logical_instance_id:
        return row.logical_instance_id
    payload = {key: getattr(row, key, None) for key in LOGICAL_ID_HASHED_FIELDS}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    row.logical_instance_id = digest
    return digest


def _allocate(total: int, keys: list[str]) -> dict[str, int]:
    quotient, remainder = divmod(total, len(keys))
    return {key: quotient + int(index < remainder) for index, key in enumerate(keys)}


def _collect_descriptors(pool: PoolsByDepth) -> set[str]:
    return {
        row.descriptor
        for by_target in pool.values()
        for rows in by_target.values()
        for row in rows
    }


def _load_source_pool(
    directory: Path,
    file_splits: tuple[SplitName, ...],
    wanted_depths: frozenset[int],
) -> PoolsByDepth:
    pool: PoolsByDepth = {}
    seen: set[str] = set()
    for file_split in file_splits:
        path = directory / f"{file_split}.jsonl"
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = ClutrrRow.model_validate_json(line)
                if _is_blocked(row.descriptor) or row.depth not in wanted_depths:
                    continue
                logical_id = _ensure_logical_id(row)
                if logical_id in seen:
                    continue
                seen.add(logical_id)
                pool.setdefault(row.depth, {}).setdefault(row.target, []).append(row)
    return pool


def _load_clean_pools(directory: Path) -> SplitPools:
    short_pool = _load_source_pool(directory, SOURCE_FILES["short"], SHORT_DEPTHS)
    test_pool = _load_source_pool(directory, SOURCE_FILES["test"], LONG_DEPTHS)

    train_pool: PoolsByDepth = {}
    valid_pool: PoolsByDepth = {}
    for depth, by_target in short_pool.items():
        for target, rows in by_target.items():
            ordered = sorted(rows, key=lambda row: row.logical_instance_id)
            validation_count = max(1, len(ordered) // 10)
            valid_pool.setdefault(depth, {})[target] = ordered[:validation_count]
            train_pool.setdefault(depth, {})[target] = ordered

    pools: SplitPools = {"train": train_pool, "valid": valid_pool, "test": test_pool}
    if _collect_descriptors(train_pool) & _collect_descriptors(test_pool):
        raise ValueError("Train and test descriptors must be length-disjoint")
    return pools


def _sample_rows(
    pool: PoolsByDepth,
    split: SplitName,
    depth: int,
    total: int,
    rng: random.Random,
) -> list[ClutrrRow]:
    by_target = pool.get(depth) or {}
    targets = sorted(target for target, rows in by_target.items() if rows)
    if not targets:
        raise ValueError(f"No usable CLUTRR graphs for {split} depth {depth}")

    rows: list[ClutrrRow] = []
    for target, count in _allocate(total, targets).items():
        source = by_target[target]
        rows.extend(copy.deepcopy(rng.choice(source)) for _ in range(count))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Balance clean CLUTRR logical graphs by depth and target"
    )
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pools = _load_clean_pools(args.graph_dir)
    rng = random.Random(args.seed)

    for split, depth_counts in ACTIVE_SPLITS.items():
        rows: list[ClutrrRow] = []
        for depth, total in depth_counts.items():
            rows.extend(_sample_rows(pools[split], split, depth, total, rng))
        rng.shuffle(rows)
        destination = args.graph_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(row.model_dump_json())
                stream.write("\n")
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
