"""Gate a guided backbone against the backbone that started its cycle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _primary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["primary"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--max-accuracy-drop", type=float, default=0.005)
    parser.add_argument("--max-cross-entropy-increase", type=float, default=0.01)
    args = parser.parse_args()

    baseline = _primary(args.baseline)
    candidate = _primary(args.candidate)
    accuracy_delta = candidate["accuracy"] - baseline["accuracy"]
    cross_entropy_delta = candidate["raw_cross_entropy"] - baseline["raw_cross_entropy"]
    print(
        f"Backbone gate: accuracy delta={accuracy_delta:+.4f}, "
        f"raw CE delta={cross_entropy_delta:+.4f}"
    )
    failures = []
    if accuracy_delta < -args.max_accuracy_drop:
        failures.append("accuracy")
    if cross_entropy_delta > args.max_cross_entropy_increase:
        failures.append("raw cross-entropy")
    if failures:
        raise SystemExit("Backbone gate failed: " + ", ".join(failures))
    print("Backbone gate passed.")


if __name__ == "__main__":
    main()
