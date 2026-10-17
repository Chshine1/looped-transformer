"""Run a YAML-defined training queue sequentially on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from looped_transformer.artifacts import get_run_output_dir
from looped_transformer.config import load_options


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def scale_learning_rates(config: dict[str, Any], factor: float) -> None:
    if not isinstance(factor, int | float) or factor <= 0:
        raise ValueError("learning_rate_scale must be positive")
    templates = config["model"]["training"]["stage_templates"]
    for template in templates.values():
        rates = template.get("optimizer", {}).get("learning_rates", {})
        for pattern in rates:
            rates[pattern] *= factor


def latest_checkpoint(run_dir: Path, stages: list[str]) -> Path | None:
    for stage in reversed(stages):
        candidates = list((run_dir / stage).glob("checkpoint-epoch-*.pt"))
        if candidates:
            return max(candidates, key=lambda path: int(path.stem.rsplit("-", 1)[1]))
    return None


def final_checkpoint(run_dir: Path, options) -> Path:
    stage_id = options.model.training.stages[-1]
    stage = options.model.training.stage_templates[stage_id]
    return run_dir / stage_id / f"checkpoint-epoch-{stage.common.epochs}.pt"


def stream_process(
    command: list[str],
    cwd: Path,
    log_path: Path,
    *,
    huggingface_offline: bool,
) -> int:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    if huggingface_offline:
        # The tokenizer is expected to have been downloaded already. These
        # prevent every queued process from waiting for an unreachable Hub.
        environment["HF_HUB_OFFLINE"] = "1"
        environment["TRANSFORMERS_OFFLINE"] = "1"
        environment["HF_HUB_DISABLE_TELEMETRY"] = "1"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.now(UTC).isoformat()}] {' '.join(command)}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", nargs="?", default="experiments/overnight.yaml")
    parser.add_argument("--only", action="append", help="run only named experiments")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force", action="store_true", help="rerun experiments already complete"
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest).resolve()
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    huggingface_offline = manifest.get("huggingface_offline", True)
    if not isinstance(huggingface_offline, bool):
        raise ValueError("huggingface_offline must be true or false")
    base_path = (manifest_path.parent / manifest["base_config"]).resolve()
    repository = base_path.parent
    os.chdir(repository)
    state_dir = (manifest_path.parent / manifest["state_directory"]).resolve()
    configs_dir = state_dir / "configs"
    logs_dir = state_dir / "logs"
    configs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    lock_path = state_dir / "queue.lock"
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Experiment queue lock exists: {lock_path}. "
            "Remove it only after confirming no queue is running."
        ) from None
    os.write(lock_fd, f"pid={os.getpid()}\n".encode())
    os.close(lock_fd)

    results_path = state_dir / "results.json"
    results: list[dict[str, Any]] = []
    selected = None if not args.only else set(args.only)
    try:
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        for experiment in manifest["experiments"]:
            name = experiment["name"]
            if not isinstance(name, str) or not re.fullmatch(
                r"[a-z0-9][a-z0-9_-]*", name
            ):
                raise ValueError(f"Invalid experiment name: {name!r}")
            if selected is not None and name not in selected:
                continue

            config = deep_merge(base, experiment.get("overrides", {}))
            if "seed" in experiment:
                config["runtime"]["seed"] = experiment["seed"]
            if "learning_rate_scale" in experiment:
                scale_learning_rates(config, experiment["learning_rate_scale"])

            config_path = configs_dir / f"{name}.yaml"
            config_path.write_text(
                yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
            )
            options = load_options(config_path)
            run_dir, run_id, _ = get_run_output_dir(
                options.runtime, options.data, options.model
            )
            complete = final_checkpoint(run_dir, options)
            resume = (
                None
                if args.force
                else latest_checkpoint(run_dir, options.model.training.stages)
            )
            if resume is None and experiment.get("resume_from") is not None:
                initial_resume = (
                    manifest_path.parent / experiment["resume_from"]
                ).resolve()
                if not initial_resume.is_file():
                    raise FileNotFoundError(
                        f"Initial checkpoint for {name!r} does not exist: "
                        f"{initial_resume}"
                    )
                resume = initial_resume
            record = {
                "name": name,
                "run_id": run_id,
                "config": str(config_path),
                "log": str(logs_dir / f"{name}.log"),
                "started_at": datetime.now(UTC).isoformat(),
            }
            results.append(record)

            if complete.is_file() and not args.force:
                record["status"] = "skipped_complete"
                print(f"[{name}] complete ({run_id}); skipping", flush=True)
                continue

            command = [
                sys.executable,
                "-m",
                "looped_transformer.main",
                "--config",
                str(config_path),
                "train",
            ]
            if resume is not None:
                command.extend(("--resume", str(resume)))
            print(
                f"[{name}] run_id={run_id}"
                + (f" resume={resume}" if resume is not None else ""),
                flush=True,
            )
            if args.dry_run:
                record["status"] = "dry_run"
                continue

            exit_code = stream_process(
                command,
                repository,
                Path(record["log"]),
                huggingface_offline=huggingface_offline,
            )
            record["exit_code"] = exit_code
            record["finished_at"] = datetime.now(UTC).isoformat()
            record["status"] = "complete" if exit_code == 0 else "failed"
            results_path.write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
            if exit_code != 0 and not manifest.get("continue_on_error", True):
                raise RuntimeError(f"Experiment {name!r} failed with {exit_code}")
    finally:
        results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        lock_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
