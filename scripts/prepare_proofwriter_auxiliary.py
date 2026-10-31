"""Prepare a length-safe, depth-4--5 ProofWriter OWA auxiliary dataset."""

import argparse
import hashlib
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from transformers.models.bert.tokenization_bert import BertTokenizerFast

SOURCE_VERSION = "V2020.12.3"
SOURCE_ROOT = f"proofwriter-dataset-{SOURCE_VERSION}/OWA/depth-5"
SOURCE_URL = (
    "https://aristo-data-public.s3-us-west-2.amazonaws.com/proofwriter/"
    f"proofwriter-dataset-{SOURCE_VERSION}.zip"
)
MODEL = "google-bert/bert-base-uncased"
REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
PROOF_DEPTHS = (4, 5)
LABELS = ("false", "true", "unknown")
FORMAT = "looped-transformer-proofwriter-owa-depth-transfer-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fingerprint(path: Path) -> dict[str, int | str]:
    return {"size": path.stat().st_size, "sha256": _sha256(path)}


def _pair_length(tokenizer: BertTokenizerFast, theory: str, query: str) -> int:
    return len(
        tokenizer.encode(
            theory,
            text_pair=query,
            add_special_tokens=True,
            truncation=False,
        )
    )


def _read_candidates(
    archive: zipfile.ZipFile,
    split: str,
    tokenizer: BertTokenizerFast,
    max_sequence_length: int,
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[dict[str, Any]]]:
    provable: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    unknown: list[dict[str, Any]] = []
    source_name = f"{SOURCE_ROOT}/meta-{split}.jsonl"
    with archive.open(source_name) as stream:
        for raw_line in stream:
            theory = json.loads(raw_line)
            for question_id, question in theory["questions"].items():
                label = str(question["answer"]).lower()
                source_depth = int(question["QDep"])
                length = _pair_length(tokenizer, theory["theory"], question["question"])
                if length > max_sequence_length:
                    continue
                example = {
                    "id": f"{theory['id']}:{question_id}",
                    "theory": theory["theory"],
                    "query": question["question"],
                    "label": label,
                    "depth": source_depth if label != "unknown" else None,
                    "source_depth": source_depth,
                    "wordpieces": length,
                }
                if label in {"true", "false"} and source_depth in PROOF_DEPTHS:
                    provable[(label, source_depth)].append(example)
                elif label == "unknown":
                    unknown.append(example)
    return provable, unknown


def _balance_provable(
    candidates: dict[tuple[str, int], list[dict[str, Any]]], rng: random.Random
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for depth in PROOF_DEPTHS:
        count = min(
            len(candidates[("true", depth)]),
            len(candidates[("false", depth)]),
        )
        for label in ("true", "false"):
            rows = list(candidates[(label, depth)])
            rng.shuffle(rows)
            selected.extend(rows[:count])
    return selected


def _length_match_unknown(
    candidates: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    count: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    # Eight-wordpiece buckets retain the long-context distribution while keeping
    # the selection deterministic and without replacement.
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        buckets[int(row["wordpieces"]) // 8].append(row)
    for rows in buckets.values():
        rng.shuffle(rows)

    target_counts = Counter(int(row["wordpieces"]) // 8 for row in reference[:count])
    selected: list[dict[str, Any]] = []
    deficits: list[int] = []
    for bucket, wanted in sorted(target_counts.items()):
        available = buckets[bucket]
        taken = min(wanted, len(available))
        selected.extend(available[:taken])
        del available[:taken]
        deficits.extend([bucket] * (wanted - taken))

    for target_bucket in deficits:
        viable = [bucket for bucket, rows in buckets.items() if rows]
        if not viable:
            raise ValueError("Not enough unknown candidates for length matching")
        nearest = min(viable, key=lambda bucket: (abs(bucket - target_bucket), bucket))
        selected.append(buckets[nearest].pop())
    if len(selected) != count:
        raise AssertionError("Unknown length matching produced the wrong row count")
    return selected


def _prepare_split(
    archive: zipfile.ZipFile,
    split: str,
    tokenizer: BertTokenizerFast,
    max_sequence_length: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{split}")
    provable_candidates, unknown_candidates = _read_candidates(
        archive, split, tokenizer, max_sequence_length
    )
    provable = _balance_provable(provable_candidates, rng)
    by_label = Counter(row["label"] for row in provable)
    count_per_label = min(by_label["true"], by_label["false"])
    if by_label["true"] != by_label["false"]:
        raise AssertionError("ProofWriter true/false balancing failed")
    reference = [row for row in provable if row["label"] == "true"]
    unknown = _length_match_unknown(unknown_candidates, reference, count_per_label, rng)
    rows = provable + unknown
    rng.shuffle(rows)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-archive",
        type=Path,
        default=Path(f"data/proofwriter-dataset-{SOURCE_VERSION}.zip"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/proofwriter-owa-d5-aux")
    )
    parser.add_argument("--max-sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.max_sequence_length <= 0:
        parser.error("--max-sequence-length must be positive")

    tokenizer = BertTokenizerFast.from_pretrained(
        MODEL, revision=REVISION, local_files_only=True
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, list[dict[str, Any]]] = {}
    with zipfile.ZipFile(args.source_archive) as archive:
        for split in ("train", "dev", "test"):
            outputs[split] = _prepare_split(
                archive,
                split,
                tokenizer,
                args.max_sequence_length,
                args.seed,
            )

    files: dict[str, Any] = {}
    for split, rows in outputs.items():
        path = args.output_dir / f"{split}.jsonl"
        _write_jsonl(path, rows)
        lengths = sorted(int(row["wordpieces"]) for row in rows)
        files[path.name] = {
            **_fingerprint(path),
            "rows": len(rows),
            "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
            "proof_depths": dict(
                sorted(
                    Counter(
                        str(row["depth"]) for row in rows if row["depth"] is not None
                    ).items()
                )
            ),
            "wordpieces": {
                "min": lengths[0],
                "p95": lengths[int(0.95 * (len(lengths) - 1))],
                "max": lengths[-1],
            },
        }

    manifest = {
        "format": FORMAT,
        "source": {
            "dataset": "ProofWriter",
            "version": SOURCE_VERSION,
            "url": SOURCE_URL,
            "archive": args.source_archive.name,
            "archive_fingerprint": _fingerprint(args.source_archive),
            "assumption": "OWA",
            "theory_partition": "depth-5",
        },
        "selection": {
            "seed": args.seed,
            "labels": list(LABELS),
            "proved_and_disproved_exact_depths": list(PROOF_DEPTHS),
            "unknown_depth": None,
            "unknown_sampling": "balanced_label_count_and_8-wordpiece-length-matched",
            "max_sequence_length": args.max_sequence_length,
            "overlength_examples_excluded": True,
        },
        "tokenizer": {"model": MODEL, "revision": REVISION},
        "files": files,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Prepared ProofWriter auxiliary data in {args.output_dir}")
    for name, details in files.items():
        print(name, details)


if __name__ == "__main__":
    main()
