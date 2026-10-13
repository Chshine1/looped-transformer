"""Run with python -m looped_transformer.analysis.warmup --help."""

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from looped_transformer.analysis.collect import collect, gradients, probes, summarize
from looped_transformer.analysis.reports import CheckpointReport, WarmupReport
from looped_transformer.artifacts import get_file_fingerprint, get_run_output_dir
from looped_transformer.config import AppOptions, load_options
from looped_transformer.config.model import ResidualScalingOptions
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.tokenizer import MNLITokenizer
from looped_transformer.types.mnli import get_logic_relation


def load_sample(
    path: str, options: AppOptions, tokenizer: MNLITokenizer, count: int, seed: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    columns = [
        options.data.premise_column,
        options.data.hypothesis_column,
        options.data.label_column,
    ]
    frame = pd.read_csv(path, sep="\t", usecols=columns, dtype=str)
    if frame[columns].isna().any().any():
        raise ValueError(f"Missing values in {path}")
    frame = frame.sample(n=min(count, len(frame)), random_state=seed).sort_index()
    encoded = tokenizer.encode(frame[columns[0]].tolist(), frame[columns[1]].tolist())
    labels = torch.tensor(
        [get_logic_relation(label).value for label in frame[columns[2]]]
    )
    return encoded.input_ids, encoded.attention_mask, labels, frame.index.tolist()


def write_report(report: WarmupReport, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "report.json"
    temporary = directory / "report.json.tmp"
    temporary.write_text(
        json.dumps(asdict(report), indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(target)
    lines = [
        "# Backbone warmup checkpoint diagnostics",
        "",
        "Dropout off; all six depths collected through the production forward. FP32, no autocast. "
        "Depths are one-based. Oracle metrics use labels and are optimistic bounds, not deployable policies.",
        "",
        "| Epoch | Scaling | CE by depth | Accuracy by depth (%) | Any-depth correct (%) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for checkpoint in report.checkpoints:
        evaluation = checkpoint.evaluation
        scaling = (
            f"gamma={checkpoint.gamma:.4f}"
            if checkpoint.gamma is not None
            else f"static={checkpoint.residual_branch_scales[0]:.5f}"
        )
        lines.append(
            f"| {checkpoint.epoch} | {scaling} | "
            + ", ".join(f"{v:.4f}" for v in evaluation.ce)
            + " | "
            + ", ".join(f"{100 * v:.2f}" for v in evaluation.accuracy)
            + f" | {100 * evaluation.any_depth_correct_accuracy:.2f} |"
        )
    lines += [
        "",
        "Full typed measurements, sample indices, source hashes, gradient matrices, probes and intervention results are in report.json.",
        "",
        "Adjacent change statistics use the same examples at every depth. Improved/worsened means a CE change exceeding 0.01. "
        "Relative representation changes are per-example norms, then averaged; classifier-visible energy is a ratio of summed squared norms in normalized pooled space. "
        "Gradient cancellation is norm(mean gradient) / mean(norm gradient), measured on the recorded fixed subset with dropout off. "
        "Interventions are inference-time distribution shifts and do not establish what retraining would achieve. "
        "Probe optimization uses a fixed budget without validation tuning; independently fitted heads have more parameters.",
        "",
    ]
    (directory / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--run", type=Path, help="Defaults to the run identified by --config"
    )
    parser.add_argument(
        "--output", type=Path, help="Defaults to reports/<run-id>/backbone_warmup"
    )
    parser.add_argument("--eval-examples", type=int, default=2048)
    parser.add_argument("--train-examples", type=int, default=4096)
    parser.add_argument("--gradient-examples", type=int, default=128)
    parser.add_argument("--intervention-examples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3, 4])
    args = parser.parse_args()
    if (
        min(
            args.eval_examples,
            args.train_examples,
            args.gradient_examples,
            args.intervention_examples,
            args.batch_size,
        )
        < 1
    ):
        parser.error("Sample and batch sizes must be positive")
    os.environ["HF_HUB_OFFLINE"] = "1"
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    options = load_options(args.config)
    if args.run is None:
        args.run = get_run_output_dir(options.runtime, options.data, options.model)[0]
    if args.output is None:
        args.output = Path("reports") / args.run.name / "backbone_warmup"
    manifest = json.loads((args.run / "artifact.json").read_text(encoding="utf-8"))
    current = asdict(options.model)
    # Training metadata omits loss/scheduler discriminators in the legacy manifest.
    for key, value in current.items():
        recorded = manifest["spec"]["model"].get(key)
        if key == "residual_scaling" and recorded is None:
            recorded = asdict(ResidualScalingOptions())
        if key != "training" and value != recorded:
            raise ValueError(f"Config differs from checkpoint manifest: model.{key}")
    tokenizer = MNLITokenizer(options.model)
    train_ids, train_mask, train_labels, train_rows = load_sample(
        str(options.data.train_file_path),
        options,
        tokenizer,
        args.train_examples,
        args.seed,
    )
    eval_ids, eval_mask, eval_labels, eval_rows = load_sample(
        str(options.data.eval_file_path),
        options,
        tokenizer,
        args.eval_examples,
        args.seed,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator().manual_seed(args.seed)
    gradient_indices = torch.randperm(len(train_labels), generator=generator)[
        : args.gradient_examples
    ]
    intervention_indices = torch.randperm(len(eval_labels), generator=generator)[
        : args.intervention_examples
    ]
    model = LoopedMNLI(options.model, tokenizer.get_tokenizer_info()).to(device)
    sources = {
        "train": get_file_fingerprint(options.data.train_file_path),
        "eval": get_file_fingerprint(options.data.eval_file_path),
    }
    for split in sources:
        if sources[split] != manifest["spec"]["data"][f"{split}_source"]:
            raise ValueError(f"{split} data differs from checkpoint manifest")
    report = WarmupReport(
        2,
        {
            "seed": args.seed,
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
            "config": str(Path(args.config).resolve()),
            "config_fingerprint": get_file_fingerprint(args.config),
            "manifest": manifest,
            "sources": sources,
            "train_rows": train_rows,
            "eval_rows": eval_rows,
            "batch_size": args.batch_size,
            "gradient_rows": [train_rows[i] for i in gradient_indices.tolist()],
            "intervention_rows": [eval_rows[i] for i in intervention_indices.tolist()],
            "compute_cost": 0.02,
            "precision": "float32, TF32 disabled",
            "source_fingerprints": {
                str(p): get_file_fingerprint(p)
                for p in sorted(Path("src/looped_transformer").rglob("*.py"))
            },
            "limitations": [
                "Historical source code is not stored in these checkpoints; results use the recorded current source.",
                "Gradient subset is a diagnostic sample, not the full training gradient.",
                "Checkpoint snapshots cannot establish the causal effect of a different training objective.",
            ],
        },
        [],
    )
    for epoch in args.epochs:
        path = args.run / "backbone_warmup" / f"checkpoint-epoch-{epoch}.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint["stage_id"] != "backbone_warmup":
            raise ValueError(f"Not a warmup checkpoint: {path}")
        model.load_state_dict(checkpoint["model"], strict=True)
        print(
            f"Epoch {epoch}: collecting evaluation ({len(eval_labels)}) and training ({len(train_labels)})",
            flush=True,
        )
        evaluation = collect(model, eval_ids, eval_mask, eval_labels, args.batch_size)
        training = collect(model, train_ids, train_mask, train_labels, args.batch_size)
        print(f"Epoch {epoch}: gradients and frozen-feature probes", flush=True)
        grad_report = gradients(
            model,
            train_ids[gradient_indices],
            train_mask[gradient_indices],
            train_labels[gradient_indices],
            args.batch_size,
        )
        probe_report = probes(model, training, evaluation)
        gamma_parameter = model._log_gamma_logit
        gamma = (
            gamma_parameter.sigmoid().item() if gamma_parameter is not None else None
        )
        interventions = {}
        if epoch == max(args.epochs):
            print(f"Epoch {epoch}: inference interventions", flush=True)
            original = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
            }
            original_static_scale = model._static_residual_scale
            scale_interventions = (
                ["gamma_plus_005", "gamma_initial_075", "gamma_one"]
                if gamma is not None
                else ["static_scale_075x", "static_scale_125x"]
            )
            for intervention in [
                "baseline_subset",
                *scale_interventions,
                "injection_off",
            ]:
                try:
                    with torch.no_grad():
                        if intervention == "gamma_one":
                            assert gamma_parameter is not None
                            gamma_parameter.fill_(30)
                        elif intervention == "gamma_plus_005":
                            assert gamma_parameter is not None and gamma is not None
                            gamma_parameter.copy_(
                                torch.tensor(
                                    min(gamma + 0.05, 0.99), device=device
                                ).logit()
                            )
                        elif intervention == "gamma_initial_075":
                            assert gamma_parameter is not None
                            gamma_parameter.copy_(
                                torch.tensor(0.75, device=device).logit()
                            )
                        elif intervention == "injection_off":
                            model._injection_gate.zero_()
                        elif intervention == "static_scale_075x":
                            model._static_residual_scale = original_static_scale * 0.75
                        elif intervention == "static_scale_125x":
                            model._static_residual_scale = original_static_scale * 1.25
                    observation = collect(
                        model,
                        eval_ids[intervention_indices],
                        eval_mask[intervention_indices],
                        eval_labels[intervention_indices],
                        args.batch_size,
                    )
                    interventions[intervention] = summarize(
                        observation, model._classifier.weight
                    )
                finally:
                    model.load_state_dict(original)
                    model._static_residual_scale = original_static_scale
        report.checkpoints.append(
            CheckpointReport(
                str(path.resolve()),
                get_file_fingerprint(path)["sha256"],
                epoch,
                checkpoint["stage_id"],
                checkpoint["global_step"],
                checkpoint["best_accuracy"],
                gamma,
                [
                    model.residual_scale(t).item()
                    for t in range(options.model.max_iterations)
                ],
                model._injection_gate.square().mean().sqrt().item(),
                summarize(evaluation, model._classifier.weight),
                summarize(training, model._classifier.weight),
                grad_report,
                probe_report,
                interventions,
                options.model.residual_scaling.mode,
                [
                    0.0,
                    *[
                        model.injection_scale(t).item()
                        for t in range(1, options.model.max_iterations)
                    ],
                ],
            )
        )
        write_report(report, args.output)
        print(f"Epoch {epoch}: saved {args.output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
