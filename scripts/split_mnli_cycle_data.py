"""Create disjoint, stratified train/dev files for alternating training cycles."""

from __future__ import annotations

import argparse
import csv
import random
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from sample_mnli_train import inspect_source, proportional_quotas


def select_dev_indices(
    source: Path,
    columns: list[str],
    strata: tuple[str, ...],
    quotas: dict[tuple[str, ...], int],
    seed: int,
    chunk_size: int,
) -> set[int]:
    rng = random.Random(seed)
    stratum_indices = tuple(columns.index(column) for column in strata)
    seen: Counter[tuple[str, ...]] = Counter()
    reservoirs: defaultdict[tuple[str, ...], list[int]] = defaultdict(list)
    offset = 0
    reader = pd.read_csv(
        source,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=chunk_size,
    )
    with reader:
        for chunk in reader:
            for local_index, row in enumerate(chunk.itertuples(index=False, name=None)):
                group = tuple(row[index] for index in stratum_indices)
                seen[group] += 1
                reservoir = reservoirs[group]
                quota = quotas[group]
                absolute_index = offset + local_index
                if len(reservoir) < quota:
                    reservoir.append(absolute_index)
                    continue
                replacement = rng.randrange(seen[group])
                if replacement < quota:
                    reservoir[replacement] = absolute_index
            offset += len(chunk)
    return {index for reservoir in reservoirs.values() for index in reservoir}


def write_split(
    source: Path,
    train_output: Path,
    dev_output: Path,
    dev_indices: set[int],
    chunk_size: int,
) -> None:
    train_output.parent.mkdir(parents=True, exist_ok=True)
    dev_output.parent.mkdir(parents=True, exist_ok=True)
    train_temporary = train_output.with_suffix(train_output.suffix + ".tmp")
    dev_temporary = dev_output.with_suffix(dev_output.suffix + ".tmp")
    offset = 0
    first = True
    try:
        reader = pd.read_csv(
            source,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            chunksize=chunk_size,
        )
        with reader:
            for chunk in reader:
                is_dev = [offset + index in dev_indices for index in range(len(chunk))]
                chunk.loc[is_dev].to_csv(
                    dev_temporary,
                    sep="\t",
                    index=False,
                    encoding="utf-8",
                    lineterminator="\n",
                    quoting=csv.QUOTE_MINIMAL,
                    mode="w" if first else "a",
                    header=first,
                )
                chunk.loc[[not value for value in is_dev]].to_csv(
                    train_temporary,
                    sep="\t",
                    index=False,
                    encoding="utf-8",
                    lineterminator="\n",
                    quoting=csv.QUOTE_MINIMAL,
                    mode="w" if first else "a",
                    header=first,
                )
                first = False
                offset += len(chunk)
        train_temporary.replace(train_output)
        dev_temporary.replace(dev_output)
    except BaseException:
        train_temporary.unlink(missing_ok=True)
        dev_temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/mnli_matched.tsv"))
    parser.add_argument(
        "--train-output", type=Path, default=Path("data/mnli_matched_final.tsv")
    )
    parser.add_argument(
        "--dev-output", type=Path, default=Path("data/mnli_cycle_dev.tsv")
    )
    parser.add_argument("--dev-size", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--stratify-by", nargs="+", default=["genre", "gold_label"])
    args = parser.parse_args()

    columns, counts = inspect_source(
        args.source, tuple(args.stratify_by), args.chunk_size
    )
    quotas = proportional_quotas(counts, args.dev_size)
    dev_indices = select_dev_indices(
        args.source,
        columns,
        tuple(args.stratify_by),
        quotas,
        args.seed,
        args.chunk_size,
    )
    write_split(
        args.source,
        args.train_output,
        args.dev_output,
        dev_indices,
        args.chunk_size,
    )
    print(
        f"Wrote {len(dev_indices)} dev rows to {args.dev_output} and the disjoint "
        f"remainder to {args.train_output}"
    )


if __name__ == "__main__":
    main()
