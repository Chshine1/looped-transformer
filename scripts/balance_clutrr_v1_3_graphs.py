"""Build the active balanced logical splits before CLUTRR text rendering."""

import argparse
import copy
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

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
ACTIVE_SPLITS = {
    "train": {2: 30_000, 3: 30_000},
    "valid": {2: 3_000, 3: 3_000},
    "test": {4: 2_000, 5: 2_000, 6: 2_000},
}


def _load_clean_pools(
    directory: Path,
) -> dict[str, dict[int, dict[str, list[dict[str, Any]]]]]:
    source_pools: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {}
    source_files = {"short": ("train", "valid"), "test": ("test",)}
    for source_split, file_splits in source_files.items():
        source_pool: dict[int, dict[str, list[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        seen: set[str] = set()
        for file_split in file_splits:
            path = directory / f"{file_split}.jsonl"
            with path.open(encoding="utf-8") as stream:
                for line in stream:
                    row = json.loads(line)
                    descriptor = row["descriptor"]
                    if any(part in descriptor for part in BLOCKED_DESCRIPTOR_FRAGMENTS):
                        continue
                    depth = len(descriptor.split(","))
                    wanted_depths = {2, 3} if source_split == "short" else {4, 5, 6}
                    if depth not in wanted_depths:
                        continue
                    logical_id = row.get("logical_instance_id")
                    if not isinstance(logical_id, str) or not logical_id:
                        logical_fields = {
                            key: row.get(key)
                            for key in (
                                "edges",
                                "source",
                                "sink",
                                "target",
                                "descriptor",
                                "rules_used",
                                "rules_used_pos",
                                "resolution_path",
                            )
                        }
                        logical_id = hashlib.sha256(
                            json.dumps(
                                logical_fields,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode()
                        ).hexdigest()
                        row["logical_instance_id"] = logical_id
                    if logical_id in seen:
                        continue
                    seen.add(logical_id)
                    source_pool[depth][row["target"]].append(row)
        source_pools[source_split] = source_pool

    pools: dict[str, dict[int, dict[str, list[dict[str, Any]]]]] = {
        split: defaultdict(lambda: defaultdict(list)) for split in ACTIVE_SPLITS
    }
    for depth, targets in source_pools["short"].items():
        for target, rows in targets.items():
            ordered = sorted(rows, key=lambda row: row["logical_instance_id"])
            validation_count = max(1, len(ordered) // 10)
            pools["valid"][depth][target].extend(ordered[:validation_count])
            pools["train"][depth][target].extend(ordered)
    pools["test"] = source_pools["test"]

    train_descriptors = {
        row["descriptor"]
        for targets in pools["train"].values()
        for rows in targets.values()
        for row in rows
    }
    test_descriptors = {
        row["descriptor"]
        for targets in pools["test"].values()
        for rows in targets.values()
        for row in rows
    }
    if train_descriptors & test_descriptors:
        raise ValueError("Train and test descriptors must be length-disjoint")
    return pools


def _allocate(total: int, keys: list[str]) -> dict[str, int]:
    quotient, remainder = divmod(total, len(keys))
    return {key: quotient + int(index < remainder) for index, key in enumerate(keys)}


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
        rows: list[dict[str, Any]] = []
        for depth, total in depth_counts.items():
            targets = sorted(
                target for target, rows in pools[split][depth].items() if rows
            )
            if not targets:
                raise ValueError(f"No usable CLUTRR graphs for {split} depth {depth}")
            for target, count in _allocate(total, targets).items():
                source = pools[split][depth][target]
                rows.extend(copy.deepcopy(rng.choice(source)) for _ in range(count))
        rng.shuffle(rows)
        destination = args.graph_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row))
                stream.write("\n")
        print(f"{split}: {len(rows)}")


if __name__ == "__main__":
    main()
