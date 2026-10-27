"""Fail unless a controller is useful and neither uniform nor over-confident."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _teacher_policy(split: dict[str, Any]) -> dict[str, Any]:
    for policy in split["policies"]:
        if policy["name"] == "teacher_accuracy_selector":
            return policy
    raise ValueError("analysis has no BCE teacher_accuracy_selector policy")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--split", default="mnli_cycle_dev")
    parser.add_argument("--min-accuracy-delta", type=float, default=-0.005)
    parser.add_argument("--min-brier-improvement", type=float, default=0.05)
    parser.add_argument("--min-effective-depths", type=float, default=3.0)
    parser.add_argument("--max-effective-depths", type=float, default=5.7)
    parser.add_argument("--min-mean-max-weight", type=float, default=0.20)
    parser.add_argument("--max-mean-max-weight", type=float, default=0.55)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    if len(analysis["epochs"]) != 1:
        raise ValueError("gate expects exactly one checkpoint pair")
    split = analysis["epochs"][0]["splits"][args.split]
    policy = _teacher_policy(split)
    guidance = policy["guidance_distribution"]
    values = {
        "accuracy delta": policy["comparison_to_final_loop"]["accuracy_difference"],
        "Brier improvement": policy["brier_improvement_over_constant"],
        "effective depths": guidance["mean_effective_depth_count"],
        "mean max weight": guidance["mean_max_weight"],
    }
    checks = {
        "accuracy delta": values["accuracy delta"] >= args.min_accuracy_delta,
        "Brier improvement": values["Brier improvement"] >= args.min_brier_improvement,
        "effective depths (lower)": values["effective depths"]
        >= args.min_effective_depths,
        "effective depths (upper)": values["effective depths"]
        <= args.max_effective_depths,
        "mean max weight (lower)": values["mean max weight"]
        >= args.min_mean_max_weight,
        "mean max weight (upper)": values["mean max weight"]
        <= args.max_mean_max_weight,
    }

    print(
        "Controller gate: "
        + ", ".join(f"{name}={value:.4f}" for name, value in values.items())
    )
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit("Controller gate failed: " + ", ".join(failed))
    print("Controller gate passed.")


if __name__ == "__main__":
    main()
