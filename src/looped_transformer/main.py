import argparse
import dataclasses
import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

import torch

from looped_transformer.analyzers.warmup_analyzer import WarmupAnalyzer
from looped_transformer.checkpoint_manager import CheckpointManagerFactory
from looped_transformer.config import load_options
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.runner import MNLIRunner
from looped_transformer.stages import build_stage_templates
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.data import DataSplit


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _find_training_artifact(checkpoint_path: Path) -> tuple[Path, dict] | None:
    for directory in checkpoint_path.parents:
        candidate = directory / "artifact.json"
        if candidate.is_file():
            return candidate, json.loads(candidate.read_text(encoding="utf-8"))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the looped MNLI transformer")
    parser.add_argument("--config", default="config.yaml")
    subparser = parser.add_subparsers(dest="command", required=True)

    train_parser = subparser.add_parser("train", help="train the model")
    train_parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="resume from a checkpoint path, or use 'latest'",
    )

    analyze_parser = subparser.add_parser("analyze", help="analyze the model")
    analyze_parser.add_argument("--checkpoint", required=True)
    analyze_parser.add_argument(
        "--output",
        help="JSON snapshot path (default: CHECKPOINT.warmup.json)",
    )
    analyze_parser.add_argument(
        "--name",
        help="label stored in the snapshot (default: checkpoint filename)",
    )
    analyze_parser.add_argument(
        "--token-samples",
        type=int,
        default=16,
        metavar="N",
        help="number of examples retaining per-token trajectories (default: 16)",
    )

    args = parser.parse_args()
    options = load_options(args.config)
    tokenizer = MNLITokenizer(options.model)
    checkpoint_manager_factory = CheckpointManagerFactory(
        options.model,
        options.data,
        options.runtime,
    )
    data_reader = MNLIDataReader(options.runtime, options.data)
    stages = build_stage_templates(options.model.training, data_reader, options.data)
    runner = MNLIRunner(
        tokenizer,
        checkpoint_manager_factory,
        data_reader,
        options.model,
        options.data,
        options.runtime,
    )

    if args.command == "train":
        runner.train(stages, args.resume)
    elif args.command == "analyze":
        if args.token_samples < 0:
            analyze_parser.error("--token-samples must be non-negative")
        # noinspection protected-member
        model, device = runner._model, runner._device

        checkpoint: dict = torch.load(
            args.checkpoint,
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(checkpoint["model"])

        if "python_random_state" in checkpoint:
            random.setstate(checkpoint["python_random_state"])

        if "torch_random_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_random_state"].cpu())

        if device.type == "cuda" and checkpoint.get("cuda_random_state"):
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_random_state"]]
            )

        labels = data_reader.get_batches_iterator(
            DataSplit.EVAL,
            shuffle=False,
        )
        checkpoint_path = Path(args.checkpoint).resolve()
        output_path = (
            Path(args.output).resolve()
            if args.output
            else checkpoint_path.with_suffix(".warmup.json")
        )
        analyzer = WarmupAnalyzer(
            model,
            options.model,
            labels,
            run_name=args.name or checkpoint_path.stem,
            max_token_samples=args.token_samples,
        )
        try:
            accuracy = runner.eval()
            snapshot = analyzer.get_report()
        finally:
            analyzer.remove_hooks()

        training_artifact = _find_training_artifact(checkpoint_path)
        snapshot["context"] = {
            "created_at": datetime.now(UTC).isoformat(),
            "checkpoint": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "checkpoint_stage": checkpoint.get("stage_id"),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_global_step": checkpoint.get("global_step"),
            "config": str(Path(args.config).resolve()),
            "options": dataclasses.asdict(options),
            "training_artifact": None
            if training_artifact is None
            else {
                "path": str(training_artifact[0].resolve()),
                "contents": training_artifact[1],
            },
            "evaluation_accuracy": accuracy,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(snapshot, indent=2, allow_nan=False),
            encoding="utf-8",
        )

        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(f"Warmup snapshot: {output_path}")


if __name__ == "__main__":
    main()
