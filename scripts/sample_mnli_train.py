"""Create a small, representative sample of the MNLI training set.

By default, the output size is the rounded mean of the matched and mismatched
validation-set sizes. Sampling is proportional to the joint distribution of
genre and gold label and is reproducible for a given seed.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path("data/mnli_train.tsv")
DEFAULT_OUTPUT = Path("data/mnli_train_sample.tsv")
DEFAULT_REFERENCES = (
    Path("data/mnli_matched.tsv"),
    Path("data/mnli_mismatched.tsv"),
)
DEFAULT_STRATA = ("genre", "gold_label")

Stratum = tuple[str, ...]


def count_rows(path: Path, chunk_size: int) -> int:
    """Count data rows without loading an entire TSV into memory."""
    # Reading one column also makes quoted newlines safe, unlike counting '\n'.
    header = pd.read_csv(path, sep="\t", nrows=0)
    if header.columns.empty:
        return 0
    first_column = str(header.columns[0])
    chunks = pd.read_csv(
        path,
        sep="\t",
        usecols=[first_column],
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )
    return sum(len(chunk) for chunk in chunks)


def proportional_quotas(counts: Counter[Stratum], size: int) -> dict[Stratum, int]:
    """Allocate an exact sample size with the largest-remainder method."""
    total = sum(counts.values())
    if total == 0:
        raise ValueError("cannot sample from an empty source")
    if size < 1:
        raise ValueError("sample size must be positive")
    if size > total:
        raise ValueError(f"sample size {size} exceeds source row count {total}")

    exact = {group: size * count / total for group, count in counts.items()}
    quotas = {group: int(value) for group, value in exact.items()}
    remaining = size - sum(quotas.values())
    ranked = sorted(
        counts,
        key=lambda group: (-(exact[group] - quotas[group]), group),
    )
    for group in ranked[:remaining]:
        quotas[group] += 1
    return quotas


def inspect_source(
    source: Path,
    strata: tuple[str, ...],
    chunk_size: int,
) -> tuple[list[str], Counter[Stratum]]:
    header = pd.read_csv(source, sep="\t", nrows=0)
    columns = [str(column) for column in header.columns]
    missing = set(strata).difference(columns)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"{source}: missing stratification columns: {names}")

    counts: Counter[Stratum] = Counter()
    chunks = pd.read_csv(
        source,
        sep="\t",
        usecols=list(strata),
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        ordered = chunk.loc[:, list(strata)]
        counts.update(tuple(row) for row in ordered.itertuples(index=False, name=None))
    return columns, counts


def stratified_reservoir_sample(
    source: Path,
    columns: list[str],
    strata: tuple[str, ...],
    quotas: dict[Stratum, int],
    seed: int,
    chunk_size: int,
) -> pd.DataFrame:
    """Uniformly sample the requested number of rows from every stratum."""
    rng = random.Random(seed)
    stratum_indices = tuple(columns.index(column) for column in strata)
    seen: Counter[Stratum] = Counter()
    reservoirs: defaultdict[Stratum, list[tuple[str, ...]]] = defaultdict(list)

    chunks = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )
    for chunk in chunks:
        for row in chunk.itertuples(index=False, name=None):
            group = tuple(row[index] for index in stratum_indices)
            seen[group] += 1
            quota = quotas[group]
            reservoir = reservoirs[group]
            if len(reservoir) < quota:
                reservoir.append(row)
                continue
            replacement = rng.randrange(seen[group])
            if replacement < quota:
                reservoir[replacement] = row

    rows = [row for reservoir in reservoirs.values() for row in reservoir]
    rng.shuffle(rows)
    return pd.DataFrame.from_records(rows, columns=columns)


def print_distribution(
    title: str,
    counts: Counter[Stratum],
    total: int,
    strata: tuple[str, ...],
) -> None:
    print(title)
    for group in sorted(counts):
        label = ", ".join(
            f"{name}={value}" for name, value in zip(strata, group, strict=True)
        )
        print(f"  {label}: {counts[group]:6d} ({counts[group] / total:7.3%})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reference",
        nargs="+",
        type=Path,
        default=list(DEFAULT_REFERENCES),
        help="files whose mean row count determines the default sample size",
    )
    parser.add_argument(
        "--size",
        type=int,
        help="sample size; defaults to the rounded mean size of --reference files",
    )
    parser.add_argument("--stratify-by", nargs="+", default=list(DEFAULT_STRATA))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size < 1:
        raise ValueError("chunk size must be positive")
    if args.source.resolve() == args.output.resolve():
        raise ValueError("source and output must be different files")

    if args.size is None:
        reference_sizes = [count_rows(path, args.chunk_size) for path in args.reference]
        if not reference_sizes:
            raise ValueError("at least one reference file is required when size is omitted")
        size = round(sum(reference_sizes) / len(reference_sizes))
        details = ", ".join(
            f"{path}={count}"
            for path, count in zip(args.reference, reference_sizes, strict=True)
        )
        print(f"Reference sizes: {details}; target={size}")
    else:
        size = args.size

    strata = tuple(args.stratify_by)
    columns, source_counts = inspect_source(args.source, strata, args.chunk_size)
    quotas = proportional_quotas(source_counts, size)
    sample = stratified_reservoir_sample(
        args.source,
        columns,
        strata,
        quotas,
        args.seed,
        args.chunk_size,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    try:
        sample.to_csv(
            temporary,
            sep="\t",
            index=False,
            encoding="utf-8",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )
        temporary.replace(args.output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    sample_counts = Counter(
        tuple(row)
        for row in sample.loc[:, list(strata)].itertuples(index=False, name=None)
    )
    print(f"Wrote {len(sample)} rows to {args.output} (seed={args.seed})")
    print_distribution("Sample distribution:", sample_counts, len(sample), strata)
    maximum_error = max(
        abs(sample_counts[group] / len(sample) - count / sum(source_counts.values()))
        for group, count in source_counts.items()
    )
    print(f"Maximum joint-stratum deviation: {maximum_error:.4%}")


if __name__ == "__main__":
    main()
