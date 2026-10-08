"""Normalize MNLI TSV files and remove rows unusable by the model.

The downloaded GLUE/MNLI files contain unescaped double quotes, so they must be
read with quoting disabled.  The normalized output uses pandas' standard TSV
quoting and can subsequently be read with an ordinary ``pd.read_csv`` call.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

TEXT_COLUMNS = ("sentence1", "sentence2")
IDENTITY_COLUMNS = ("index", "promptID", "pairID", "genre")
VALID_LABELS = frozenset({"entailment", "contradiction", "neutral"})
MISSING_MARKERS = frozenset({"", "n/a", "na", "nan", "none", "null"})


@dataclass
class CleaningReport:
    input_rows: int = 0
    output_rows: int = 0
    removed: Counter[str] = field(default_factory=Counter)
    examples: dict[str, list[str]] = field(default_factory=dict)

    def reject(self, reason: str, indices: pd.Series) -> None:
        if indices.empty:
            return
        self.removed[reason] += len(indices)
        samples = self.examples.setdefault(reason, [])
        samples.extend(str(value) for value in indices.head(5 - len(samples)))


def missing_marker(values: pd.Series) -> pd.Series:
    """Find empty and textual NA values without letting pandas coerce them."""
    normalized = values.astype("string").str.strip().str.casefold()
    return values.isna() | normalized.isin(MISSING_MARKERS)


def clean_file(source: Path, destination: Path, chunk_size: int) -> CleaningReport:
    # noinspection argument-list
    header = pd.read_csv(
        source,
        sep="\t",
        quoting=csv.QUOTE_NONE,
        keep_default_na=False,
        nrows=0,
    )
    required = {*IDENTITY_COLUMNS, *TEXT_COLUMNS}
    missing_columns = required.difference(header.columns)
    if missing_columns:
        names = ", ".join(sorted(missing_columns))
        raise ValueError(f"{source}: missing required columns: {names}")

    report = CleaningReport()
    seen_indices: set[str] = set()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    wrote_header = False

    def skip_bad_line(fields: list[str]) -> None:
        report.input_rows += 1
        report.removed["wrong field count"] += 1
        samples = report.examples.setdefault("wrong field count", [])
        if len(samples) < 5:
            samples.append(fields[0] if fields else "<empty>")
        return None

    try:
        # noinspection argument-list
        chunks = pd.read_csv(
            source,
            sep="\t",
            quoting=csv.QUOTE_NONE,
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
            engine="python",
            on_bad_lines=skip_bad_line,
        )
        for chunk in chunks:
            report.input_rows += len(chunk)
            rejected = pd.Series(False, index=chunk.index)

            malformed = chunk.isna().any(axis="columns")
            report.reject("wrong field count", chunk.loc[malformed, "index"])
            rejected |= malformed

            duplicate_header = chunk["index"].eq("index")
            report.reject("repeated header", chunk.loc[duplicate_header, "index"])
            rejected |= duplicate_header

            invalid_index = pd.to_numeric(chunk["index"], errors="coerce").isna()
            invalid_index &= ~duplicate_header
            report.reject("invalid index", chunk.loc[invalid_index, "index"])
            rejected |= invalid_index

            duplicate_index = (
                chunk["index"].isin(seen_indices) | chunk["index"].duplicated()
            )
            duplicate_index &= ~rejected
            report.reject("duplicate index", chunk.loc[duplicate_index, "index"])
            rejected |= duplicate_index

            for column in TEXT_COLUMNS:
                invalid_text = missing_marker(chunk[column]) & ~rejected
                report.reject(f"missing {column}", chunk.loc[invalid_text, "index"])
                rejected |= invalid_text

            if "gold_label" in chunk:
                normalized_labels = chunk["gold_label"].str.strip().str.casefold()
                invalid_label = ~normalized_labels.isin(VALID_LABELS) & ~rejected
                report.reject("invalid gold_label", chunk.loc[invalid_label, "index"])
                rejected |= invalid_label
                chunk.loc[:, "gold_label"] = normalized_labels

            accepted = chunk.loc[~rejected]
            seen_indices.update(accepted["index"])
            accepted.to_csv(
                temporary,
                sep="\t",
                index=False,
                mode="a" if wrote_header else "w",
                header=not wrote_header,
                encoding="utf-8",
                lineterminator="\n",
            )
            wrote_header = True
            report.output_rows += len(accepted)

        if not wrote_header:
            header.to_csv(temporary, sep="\t", index=False, encoding="utf-8")
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean MNLI TSVs using the same pandas parser as training"
    )
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument("--chunk-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for source in args.inputs:
        destination = args.output_dir / source.name
        report = clean_file(source, destination, args.chunk_size)
        print(f"{source} -> {destination}")
        print(f"  rows: {report.input_rows} -> {report.output_rows}")
        for reason, count in report.removed.items():
            examples = ", ".join(report.examples[reason])
            print(f"  removed {count} ({reason}); examples: {examples}")
        # noinspection argument-list
        columns = pd.read_csv(destination, sep="\t", nrows=0).columns
        if "gold_label" not in columns:
            print("  note: unlabeled test data; not usable for evaluation")


if __name__ == "__main__":
    main()
