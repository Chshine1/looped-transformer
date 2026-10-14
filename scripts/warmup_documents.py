"""Compare saved WarmupAnalyzer snapshots without importing PyTorch.

This tool intentionally produces reusable figures and tabular data rather than a
fixed report. A paper, notebook, or hand-written TeX document can choose from the
generated artifacts and cite their entries in ``index.json``.
"""

import argparse
import csv
import json
import math
import re
from itertools import combinations
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use("Agg", force=True)


DEFAULT_METRICS = (
    "ce",
    "ce_abs_change",
    "loop_delta",
    "loop_cos",
    "aligned_delta",
    "aligned_cos",
    "pooled_delta",
    "pooling_residual_fraction",
    "pooling_cancelled_motion_fraction",
    "head_invisible_state_fraction",
    "head_invisible_motion_fraction",
    "ce_tangent_motion_fraction",
    "pre_rotation_delta",
    "rotation_delta",
    "rotation_norm_error",
    "probability_delta",
    "accuracy",
)

TOKEN_METRICS = (
    "loop_norm",
    "loop_delta",
    "loop_cos",
    "aligned_delta",
    "aligned_cos",
    "pooling_residual_norm",
)


def parse_input(value: str) -> tuple[str | None, Path]:
    if "=" not in value:
        return None, Path(value)
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("input must be PATH or LABEL=PATH")
    return label, Path(path)


def load_snapshots(specifications: list[tuple[str | None, Path]]) -> dict[str, dict]:
    snapshots = {}
    for requested_label, path in specifications:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        if snapshot.get("schema_version") != 2 or not isinstance(
            snapshot.get("loops"), list
        ):
            raise ValueError(f"{path} is not a WarmupAnalyzer schema-v2 snapshot")
        label = requested_label or snapshot.get("run_name") or path.stem
        if label in snapshots:
            raise ValueError(f"Duplicate run label: {label!r}")
        snapshots[label] = {"source": path.resolve(), "data": snapshot}
    return snapshots


def available_metrics(snapshots: dict[str, dict]) -> list[str]:
    return sorted(
        {
            metric
            for item in snapshots.values()
            for loop in item["data"]["loops"]
            for metric in loop["metrics"]
        }
    )


def safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-.")
    return normalized or "metric"


def numeric(values):
    return [math.nan if value is None else value for value in values]


def write_aggregate_csv(path: Path, snapshots: dict[str, dict]) -> None:
    fields = (
        "run",
        "source",
        "iteration",
        "t",
        "observed_examples",
        "metric",
        "mean",
        "std",
        "p10",
        "median",
        "p90",
        "metric_count",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for run, item in snapshots.items():
            for loop in item["data"]["loops"]:
                for metric, statistics in loop["metrics"].items():
                    writer.writerow(
                        {
                            "run": run,
                            "source": item["source"],
                            "iteration": loop["iteration"],
                            "t": loop["t"],
                            "observed_examples": loop["count"],
                            "metric": metric,
                            **{
                                key: statistics.get(key)
                                for key in ("mean", "std", "p10", "median", "p90")
                            },
                            "metric_count": statistics.get("count"),
                        }
                    )


def plot_aggregate_metric(
    metric: str,
    snapshots: dict[str, dict],
    figures_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    figure, axis = plt.subplots(figsize=(8.5, 4.8), layout="constrained")
    plotted = False
    for run, item in snapshots.items():
        loops = item["data"]["loops"]
        x = [loop["iteration"] for loop in loops]
        statistics = [loop["metrics"].get(metric, {}) for loop in loops]
        means = [entry.get("mean") for entry in statistics]
        if all(value is None for value in means):
            continue
        (line,) = axis.plot(x, numeric(means), marker="o", label=run)
        axis.fill_between(
            x,
            numeric([entry.get("p10") for entry in statistics]),
            numeric([entry.get("p90") for entry in statistics]),
            color=line.get_color(),
            alpha=0.14,
        )
        plotted = True
    if not plotted:
        plt.close(figure)
        return []
    axis.set(xlabel="Iteration (1-based)", ylabel=metric, title=metric)
    axis.grid(alpha=0.2)
    axis.legend()
    outputs = []
    for suffix in formats:
        output = figures_dir / f"aggregate-{safe_name(metric)}.{suffix}"
        figure.savefig(output, dpi=180, bbox_inches="tight")
        outputs.append(output.resolve())
    plt.close(figure)
    return outputs


def plot_token_sample(
    run: str,
    snapshot: dict,
    sample_id: int,
    figures_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    token_samples = snapshot["token_samples"]
    rows = token_samples.get(str(sample_id), token_samples.get(sample_id))
    if rows is None:
        raise ValueError(f"Run {run!r} has no retained token sample {sample_id}")
    rows = {int(t): row for t, row in rows.items()}
    length = len(snapshot["loops"])
    figure, axes = plt.subplots(3, 2, figsize=(11, 10), layout="constrained")
    positions = sorted(
        {position for row in rows.values() for position in row["positions"]}
    )
    for axis, metric in zip(axes.flat, TOKEN_METRICS, strict=True):
        for position in positions:
            values = []
            for t in range(length):
                row = rows.get(t, {})
                row_positions = row.get("positions", [])
                value = (
                    row[metric][row_positions.index(position)]
                    if position in row_positions and metric in row
                    else None
                )
                values.append(value)
            axis.plot(
                range(1, length + 1),
                numeric(values),
                alpha=0.55,
                linewidth=0.8,
            )
        axis.set(title=metric, xlabel="Iteration (1-based)")
        axis.grid(alpha=0.15)
    figure.suptitle(f"{run}: token trajectories for sample {sample_id}")
    outputs = []
    basename = f"tokens-{safe_name(run)}-sample-{sample_id}"
    for suffix in formats:
        output = figures_dir / f"{basename}.{suffix}"
        figure.savefig(output, dpi=180, bbox_inches="tight")
        outputs.append(output.resolve())
    plt.close(figure)
    return outputs


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build reusable tables and figures from warmup JSON snapshots"
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=parse_input,
        metavar="[LABEL=]PATH",
        help="repeat for each checkpoint/configuration to compare",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--metric",
        action="append",
        help="aggregate metric to plot; repeat as needed (default: curated set)",
    )
    parser.add_argument(
        "--all-metrics",
        action="store_true",
        help="plot every aggregate metric found in the input snapshots",
    )
    parser.add_argument(
        "--token-sample",
        action="append",
        type=int,
        default=[],
        help="retained sample ID to plot for every run; repeat as needed",
    )
    parser.add_argument(
        "--format",
        action="append",
        choices=("pdf", "png", "svg"),
        dest="formats",
        help="figure format; repeat as needed (default: pdf)",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="print available aggregate metrics before generating outputs",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    snapshots = load_snapshots(arguments.input)
    metrics_available = available_metrics(snapshots)
    if arguments.list_metrics:
        print("\n".join(metrics_available))
    requested_metrics = (
        metrics_available
        if arguments.all_metrics
        else arguments.metric or list(DEFAULT_METRICS)
    )
    unknown = sorted(set(requested_metrics) - set(metrics_available))
    if arguments.metric and unknown:
        raise ValueError("Unknown metrics: " + ", ".join(unknown))
    metrics = [metric for metric in requested_metrics if metric in metrics_available]
    formats = tuple(dict.fromkeys(arguments.formats or ["pdf"]))
    output_dir = arguments.output_dir.resolve()
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = output_dir / "aggregate_metrics.csv"
    write_aggregate_csv(aggregate_csv, snapshots)

    figures = {}
    for metric in metrics:
        paths = plot_aggregate_metric(metric, snapshots, figures_dir, formats)
        if paths:
            figures[f"aggregate:{metric}"] = [str(path) for path in paths]
    for run, item in snapshots.items():
        for sample_id in arguments.token_sample:
            paths = plot_token_sample(
                run, item["data"], sample_id, figures_dir, formats
            )
            figures[f"tokens:{run}:{sample_id}"] = [str(path) for path in paths]

    comparisons = []
    for (left_name, left), (right_name, right) in combinations(snapshots.items(), 2):
        left_data, right_data = left["data"], right["data"]
        left_keys = set(left_data.get("sample_keys", {}).values())
        right_keys = set(right_data.get("sample_keys", {}).values())
        left_context = left_data.get("context", {})
        right_context = right_data.get("context", {})
        left_artifact = left_context.get("training_artifact") or {}
        right_artifact = right_context.get("training_artifact") or {}
        left_model = left_artifact.get("contents", {}).get("spec", {}).get(
            "model"
        ) or left_context.get("options", {}).get("model")
        right_model = right_artifact.get("contents", {}).get("spec", {}).get(
            "model"
        ) or right_context.get("options", {}).get("model")
        comparisons.append(
            {
                "left": left_name,
                "right": right_name,
                "same_ordered_samples": bool(left_keys)
                and left_data.get("ordered_sample_fingerprint")
                == right_data.get("ordered_sample_fingerprint"),
                "shared_sample_keys": len(left_keys & right_keys),
                "left_sample_keys": len(left_keys),
                "right_sample_keys": len(right_keys),
                "same_model_configuration": left_model == right_model,
            }
        )

    index = {
        "schema_version": 1,
        "inputs": {
            run: {
                "source": str(item["source"]),
                "context": item["data"].get("context"),
                "sample_count": item["data"]["sample_count"],
                "ordered_sample_fingerprint": item["data"].get(
                    "ordered_sample_fingerprint"
                ),
                "max_iterations": len(item["data"]["loops"]),
                "classifier_visible_rank": item["data"].get("classifier_visible_rank"),
                "plateau": item["data"].get("plateau"),
                "measurement": item["data"].get("measurement"),
            }
            for run, item in snapshots.items()
        },
        "available_metrics": metrics_available,
        "plotted_metrics": metrics,
        "comparisons": comparisons,
        "aggregate_csv": str(aggregate_csv.resolve()),
        "figures": figures,
    }
    index_path = output_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Artifact index: {index_path}")


if __name__ == "__main__":
    main()
