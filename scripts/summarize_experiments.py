"""Summarize per-loop validation metrics from an overnight experiment queue."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

METRIC_PATTERN = re.compile(
    r"epoch=(?P<epoch>\d+) validation_accuracy=(?P<dynamic>[\d.]+)% "
    r"validation_loop_ce=\[(?P<ce>[^]]+)] "
    r"validation_loop_accuracy=\[(?P<accuracy>[^]]+)]"
)


def parse_log(path: Path) -> list[dict]:
    rows = []
    for match in METRIC_PATTERN.finditer(path.read_text(encoding="utf-8")):
        ce = [float(value) for value in match.group("ce").split(",")]
        accuracy = [
            float(value.removesuffix("%")) / 100
            for value in match.group("accuracy").split(",")
        ]
        rows.append(
            {
                "name": path.stem,
                "epoch": int(match.group("epoch")),
                "dynamic_accuracy": float(match.group("dynamic")) / 100,
                "loop_ce": ce,
                "loop_accuracy": accuracy,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state_directory", nargs="?", default="outputs/overnight")
    args = parser.parse_args()
    state_dir = Path(args.state_directory)
    rows = [
        row
        for path in sorted((state_dir / "logs").glob("*.log"))
        for row in parse_log(path)
    ]
    if not rows:
        raise ValueError(f"No completed validation records found under {state_dir}")

    csv_path = state_dir / "metrics.csv"
    loop_count = len(rows[0]["loop_ce"])
    fieldnames = ["name", "epoch", "dynamic_accuracy"] + [
        f"{metric}_loop_{loop}"
        for metric in ("ce", "accuracy")
        for loop in range(1, loop_count + 1)
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in ("name", "epoch", "dynamic_accuracy")}
            for metric in ("ce", "accuracy"):
                for loop, value in enumerate(row[f"loop_{metric}"], 1):
                    flat[f"{metric}_loop_{loop}"] = value
            writer.writerow(flat)

    best_rows = []
    for name in sorted({row["name"] for row in rows}):
        run_rows = [row for row in rows if row["name"] == name]
        best = min(run_rows, key=lambda row: row["loop_ce"][-1])
        best_rows.append(best)
    best_rows.sort(key=lambda row: row["loop_ce"][-1])

    markdown = [
        "# Overnight experiment summary",
        "",
        "Rows select each run's epoch with the lowest final-loop validation CE.",
        "",
        "| Experiment | Epoch | Final CE | Final accuracy | Dynamic accuracy | CE 4→6 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in best_rows:
        tail_ce = " → ".join(f"{value:.4f}" for value in row["loop_ce"][-3:])
        markdown.append(
            f"| {row['name']} | {row['epoch']} | {row['loop_ce'][-1]:.4f} | "
            f"{row['loop_accuracy'][-1]:.2%} | {row['dynamic_accuracy']:.2%} | "
            f"{tail_ce} |"
        )
    summary_path = state_dir / "summary.md"
    summary_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {summary_path}")


if __name__ == "__main__":
    main()
