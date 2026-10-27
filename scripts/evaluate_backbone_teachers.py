"""Evaluate paired MNLI backbone and controller-teacher checkpoints.

The teacher is intentionally described as a trajectory selector: its bidirectional
GRU consumes every loop state, so these metrics must not be interpreted as online
early-exit compute savings.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import Tensor
from torch.nn import functional

from looped_transformer.app.config import load_options
from looped_transformer.domains.controller_teacher.controller_teacher import (
    ControllerTeacher,
    ControllerTeacherOptions,
)
from looped_transformer.domains.controller_teacher.controller_teacher_training import (
    build_loop_quality_targets,
)
from looped_transformer.domains.guided_backbone.model import (
    GuidanceWeightOptions,
    build_guidance_weights,
)
from looped_transformer.domains.mnli.modules.model import LoopedMNLI
from looped_transformer.domains.mnli.services.tokenizer import MNLITokenizer
from looped_transformer.domains.mnli.types.logic_relation import LogicRelation


@dataclass(frozen=True)
class CheckpointPair:
    epoch: int
    backbone: Path
    teacher: Path


def _checkpoint_pairs(output_root: Path) -> list[CheckpointPair]:
    backbones = {
        int(match.group(1)): path
        for path in output_root.glob(
            "backbone-rerun/*/backbone_warmup/checkpoint-epoch-*.pt"
        )
        if (match := re.search(r"checkpoint-epoch-(\d+)\.pt$", path.name))
    }
    pairs: list[CheckpointPair] = []
    for directory in output_root.glob("controller-teacher-from-backbone-epoch-*"):
        match = re.search(r"epoch-(\d+)$", directory.name)
        candidates = list(directory.glob("*/controller_teacher/checkpoint-epoch-1.pt"))
        smoothed_bce_selectors = [
            candidate
            for candidate in candidates
            if _teacher_loss_type(candidate) == "smoothed_per_loop_correctness_bce"
        ]
        soft_selectors = [
            candidate
            for candidate in candidates
            if _teacher_loss_type(candidate)
            in {
                "soft_loop_quality_kl",
                "soft_loop_quality_kl_v2",
                "normalized_correct_loop_kl",
            }
        ]
        accuracy_selectors = [
            candidate
            for candidate in candidates
            if _teacher_loss_type(candidate) == "per_loop_correctness_bce"
        ]
        eligible = (
            smoothed_bce_selectors or soft_selectors or accuracy_selectors or candidates
        )
        teacher = max(eligible, key=lambda path: path.stat().st_mtime, default=None)
        if match and teacher is not None and int(match.group(1)) in backbones:
            epoch = int(match.group(1))
            pairs.append(CheckpointPair(epoch, backbones[epoch], teacher))
    return sorted(pairs, key=lambda pair: pair.epoch)


def _teacher_loss_type(checkpoint: Path) -> str | None:
    return _teacher_loss_options(checkpoint).get("type")


def _teacher_loss_options(checkpoint: Path) -> dict[str, Any]:
    artifact_path = checkpoint.parents[1] / "artifact.json"
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        options = artifact["spec"]["stage"]["loss_options"]
        return options if isinstance(options, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return {}


def _load_model(model: torch.nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    model.eval()


def _label_tensor(labels: list[str], device: torch.device) -> Tensor:
    return torch.tensor(
        [LogicRelation.from_string(label).value for label in labels],
        dtype=torch.long,
        device=device,
    )


def _capture_aligned_state(
    storage: list[Tensor],
    _module: torch.nn.Module,
    _inputs: tuple[Any, ...],
    output: Tensor,
) -> None:
    storage.append(output)


def collect_outputs(
    backbone: LoopedMNLI,
    teacher: ControllerTeacher,
    tokenizer: MNLITokenizer,
    data_file: Path,
    batch_size: int,
    device: torch.device,
    max_examples: int | None,
) -> dict[str, Tensor]:
    frame = pd.read_csv(
        data_file,
        sep="\t",
        dtype=str,
        usecols=["sentence1", "sentence2", "gold_label"],
    )
    if max_examples is not None:
        frame = frame.iloc[:max_examples]

    all_logits: list[Tensor] = []
    all_teacher_logits: list[Tensor] = []
    all_halt_probabilities: list[Tensor] = []
    all_labels: list[Tensor] = []
    started = time.monotonic()
    total = len(frame)

    for start in range(0, total, batch_size):
        batch = frame.iloc[start : start + batch_size]
        encoding = tokenizer.encode(
            batch["sentence1"].tolist(), batch["sentence2"].tolist()
        )
        input_ids = encoding.input_ids.to(device)
        padding_mask = encoding.attention_mask.to(device)
        labels = _label_tensor(batch["gold_label"].tolist(), device)
        aligned_states: list[Tensor] = []

        handle = backbone._alignment_rotation.register_forward_hook(
            functools.partial(_capture_aligned_state, aligned_states)
        )
        try:
            with torch.inference_mode():
                output = backbone(input_ids, padding_mask)
                states = torch.stack(aligned_states)
                teacher_logits = teacher(states, padding_mask)
        finally:
            handle.remove()

        all_logits.append(output["timed_logits"].float().cpu())
        all_halt_probabilities.append(output["halt_probabilities"].float().cpu())
        all_teacher_logits.append(teacher_logits.float().cpu())
        all_labels.append(labels.cpu())

        done = min(start + batch_size, total)
        if done == total or done % max(batch_size, 512) < batch_size:
            elapsed = time.monotonic() - started
            print(
                f"  {data_file.stem}: {done}/{total} examples "
                f"({done / max(elapsed, 1e-6):.1f}/s)",
                flush=True,
            )

    # Batch is dimension 1 for all time-major outputs.
    return {
        "logits": torch.cat(all_logits, dim=1),
        "teacher_logits": torch.cat(all_teacher_logits, dim=1),
        "halt_probabilities": torch.cat(all_halt_probabilities, dim=1),
        "labels": torch.cat(all_labels),
    }


def _selected(values: Tensor, choices: Tensor) -> Tensor:
    return values.gather(0, choices.unsqueeze(0)).squeeze(0)


def _ece(probabilities: Tensor, correct: Tensor, bins: int = 15) -> float:
    confidence, _ = probabilities.max(dim=-1)
    total = max(1, confidence.numel())
    result = 0.0
    edges = torch.linspace(0, 1, bins + 1)
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            result += (
                mask.sum().item()
                / total
                * abs(
                    correct[mask].float().mean().item() - confidence[mask].mean().item()
                )
            )
    return result


def _policy_metrics(
    name: str,
    choices: Tensor,
    ce: Tensor,
    correct: Tensor,
    cost: Tensor,
    oracle_choices: Tensor,
) -> dict[str, Any]:
    count = choices.numel()
    chosen_correct = _selected(correct.float(), choices)
    accuracy = chosen_correct.mean().item()
    # Wilson score interval remains well behaved near zero and one.
    z = 1.959963984540054
    denominator = 1 + z**2 / count
    center = (accuracy + z**2 / (2 * count)) / denominator
    radius = (
        z
        * math.sqrt(accuracy * (1 - accuracy) / count + z**2 / (4 * count**2))
        / denominator
    )
    histogram = torch.bincount(choices, minlength=ce.shape[0])
    return {
        "name": name,
        "accuracy": accuracy,
        "accuracy_wilson_95": [center - radius, center + radius],
        "cross_entropy": _selected(ce, choices).mean().item(),
        "objective_cost": _selected(cost, choices).mean().item(),
        "mean_loop": (choices.float() + 1).mean().item(),
        "loop_histogram": histogram.tolist(),
        "oracle_choice_agreement": (choices == oracle_choices).float().mean().item(),
        "count": count,
    }


def analyse(
    outputs: dict[str, Tensor],
    incorrect_cost: float = 0.5,
    teacher_loss_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    logits = outputs["logits"]
    labels = outputs["labels"]
    loops, count, classes = logits.shape
    repeated_labels = labels.repeat(loops)
    ce = functional.cross_entropy(
        logits.reshape(-1, classes), repeated_labels, reduction="none"
    ).reshape(loops, count)
    probabilities = logits.softmax(dim=-1)
    predictions = logits.argmax(dim=-1)
    correct = predictions.eq(labels.unsqueeze(0))
    cost = ce + incorrect_cost * (~correct).float()
    oracle_choices = cost.argmin(dim=0)

    shuffle_generator = torch.Generator().manual_seed(1729)
    shuffled_accuracy_gains: list[float] = []
    shuffled_cost_reductions: list[float] = []
    for _ in range(64):
        shuffled_labels = labels[torch.randperm(count, generator=shuffle_generator)]
        shuffled_repeated = shuffled_labels.repeat(loops)
        shuffled_ce = functional.cross_entropy(
            logits.reshape(-1, classes), shuffled_repeated, reduction="none"
        ).reshape(loops, count)
        shuffled_correct = predictions.eq(shuffled_labels.unsqueeze(0))
        shuffled_cost = shuffled_ce + incorrect_cost * (~shuffled_correct).float()
        shuffled_oracle = shuffled_cost.argmin(dim=0)
        shuffled_best_fixed = int(shuffled_cost.mean(dim=1).argmin().item())
        shuffled_accuracy_gains.append(
            (
                _selected(shuffled_correct.float(), shuffled_oracle).mean()
                - shuffled_correct[shuffled_best_fixed].float().mean()
            ).item()
        )
        shuffled_cost_reductions.append(
            (
                shuffled_cost[shuffled_best_fixed].mean()
                - _selected(shuffled_cost, shuffled_oracle).mean()
            ).item()
        )

    policies: list[dict[str, Any]] = []
    fixed_costs = cost.mean(dim=1)
    best_fixed = int(fixed_costs.argmin().item())
    for loop in range(loops):
        choices = torch.full((count,), loop, dtype=torch.long)
        policies.append(
            _policy_metrics(
                f"fixed_loop_{loop + 1}", choices, ce, correct, cost, oracle_choices
            )
        )

    policies.append(
        _policy_metrics(
            "per_example_oracle", oracle_choices, ce, correct, cost, oracle_choices
        )
    )

    teacher_loss_options = teacher_loss_options or {}
    teacher_loss_type = teacher_loss_options.get("type")
    bce_loss_types = {
        "per_loop_correctness_bce",
        "smoothed_per_loop_correctness_bce",
    }
    if teacher_loss_type in bce_loss_types:
        teacher_scores = outputs["teacher_logits"].sigmoid()
        teacher_name = "teacher_accuracy_selector"
    else:
        teacher_scores = outputs["teacher_logits"].softmax(dim=0)
        teacher_name = "teacher_soft_loop_selector"
    teacher_choices = teacher_scores.argmax(dim=0)
    teacher_hard = _policy_metrics(
        teacher_name,
        teacher_choices,
        ce,
        correct,
        cost,
        oracle_choices,
    )
    teacher_hard["selected_score"] = (
        _selected(teacher_scores, teacher_choices).mean().item()
    )
    if teacher_loss_type in bce_loss_types:
        brier = (teacher_scores - correct.float()).square().mean().item()
        loop_base_rates = correct.float().mean(dim=1, keepdim=True)
        baseline_brier = (loop_base_rates - correct.float()).square().mean().item()
        guidance_options = GuidanceWeightOptions()
        guidance_weights = build_guidance_weights(
            outputs["teacher_logits"], guidance_options
        )
        guidance_entropy = -(
            guidance_weights * guidance_weights.clamp_min(1e-12).log()
        ).sum(dim=0)
        teacher_hard["per_loop_correctness_brier"] = brier
        teacher_hard["per_loop_constant_baseline_brier"] = baseline_brier
        teacher_hard["brier_improvement_over_constant"] = (
            baseline_brier - brier
        ) / baseline_brier
        teacher_hard["guidance_distribution"] = {
            "options": guidance_options.model_dump(mode="json"),
            "mean_effective_depth_count": guidance_entropy.exp().mean().item(),
            "mean_max_weight": guidance_weights.max(dim=0).values.mean().item(),
            "mean_distribution": guidance_weights.mean(dim=1).tolist(),
        }
    else:
        entropy = -(teacher_scores * teacher_scores.clamp_min(1e-12).log()).sum(dim=0)
        teacher_hard["mean_depth_entropy"] = entropy.mean().item()
        teacher_hard["mean_effective_depth_count"] = entropy.exp().mean().item()
        if teacher_loss_type in {
            "soft_loop_quality_kl",
            "soft_loop_quality_kl_v2",
            "normalized_correct_loop_kl",
        }:
            target_distribution = build_loop_quality_targets(
                logits,
                labels,
                temperature=float(teacher_loss_options["target_temperature"]),
                neighbor_smoothing=float(teacher_loss_options["neighbor_smoothing"]),
                uniform_smoothing=float(teacher_loss_options["uniform_smoothing"]),
            )
            target_entropy = -(
                target_distribution * target_distribution.clamp_min(1e-12).log()
            ).sum(dim=0)
            target_choices = target_distribution.argmax(dim=0)
            target_mean = target_distribution.mean(dim=1, keepdim=True)
            target_to_prediction_kl = functional.kl_div(
                teacher_scores.clamp_min(1e-12).log(),
                target_distribution,
                reduction="none",
            ).sum(dim=0)
            target_to_population_prior_kl = functional.kl_div(
                target_mean.clamp_min(1e-12).log().expand_as(target_distribution),
                target_distribution,
                reduction="none",
            ).sum(dim=0)
            teacher_hard["soft_target_diagnostics"] = {
                "target_to_prediction_kl": target_to_prediction_kl.mean().item(),
                "target_to_population_prior_kl": (
                    target_to_population_prior_kl.mean().item()
                ),
                "mean_target_effective_depth_count": target_entropy.exp().mean().item(),
                "mean_target_distribution": target_distribution.mean(dim=1).tolist(),
                "mean_predicted_distribution": teacher_scores.mean(dim=1).tolist(),
                "target_argmax_histogram": torch.bincount(
                    target_choices, minlength=loops
                ).tolist(),
                "target_argmax_accuracy": _selected(correct.float(), target_choices)
                .mean()
                .item(),
                "prediction_target_argmax_agreement": (
                    (teacher_choices == target_choices).float().mean().item()
                ),
            }
    teacher_correct = _selected(correct.float(), teacher_choices).bool()
    final_correct = correct[-1]
    paired_delta = teacher_correct.float() - final_correct.float()
    paired_standard_error = paired_delta.std(unbiased=True).item() / math.sqrt(count)
    teacher_hard["comparison_to_final_loop"] = {
        "accuracy_difference": paired_delta.mean().item(),
        "accuracy_difference_normal_95": [
            paired_delta.mean().item() - 1.959963984540054 * paired_standard_error,
            paired_delta.mean().item() + 1.959963984540054 * paired_standard_error,
        ],
        "rescued_final_errors": ((~final_correct) & teacher_correct).sum().item(),
        "harmed_final_correct": (final_correct & (~teacher_correct)).sum().item(),
        "same_correctness": (final_correct == teacher_correct).sum().item(),
    }
    policies.append(teacher_hard)

    halt_probabilities = outputs["halt_probabilities"]
    halt_choices = halt_probabilities.argmax(dim=0)
    halt = _policy_metrics(
        "frozen_halting_head_argmax",
        halt_choices,
        ce,
        correct,
        cost,
        oracle_choices,
    )
    halt["expected_objective_cost"] = (
        (halt_probabilities * cost).sum(dim=0).mean().item()
    )
    policies.append(halt)

    confidence = probabilities.max(dim=-1).values
    for threshold in (0.50, 0.60, 0.70, 0.80, 0.90):
        qualifies = confidence >= threshold
        choices = torch.full((count,), loops - 1, dtype=torch.long)
        has_choice = qualifies.any(dim=0)
        choices[has_choice] = qualifies.float().argmax(dim=0)[has_choice]
        policies.append(
            _policy_metrics(
                f"causal_confidence_{threshold:.2f}",
                choices,
                ce,
                correct,
                cost,
                oracle_choices,
            )
        )

    best_fixed_cost = fixed_costs[best_fixed].item()
    oracle_cost = _selected(cost, oracle_choices).mean().item()
    oracle_accuracy = _selected(correct.float(), oracle_choices).mean().item()
    best_fixed_accuracy = correct[best_fixed].float().mean().item()
    denominator = best_fixed_cost - oracle_cost
    for policy in policies:
        policy["cost_regret_to_oracle"] = policy["objective_cost"] - oracle_cost
        policy["headroom_captured_vs_best_fixed"] = (
            (best_fixed_cost - policy["objective_cost"]) / denominator
            if denominator > 1e-12
            else math.nan
        )

    per_loop = []
    for loop in range(loops):
        per_loop.append(
            {
                "loop": loop + 1,
                "accuracy": correct[loop].float().mean().item(),
                "cross_entropy": ce[loop].mean().item(),
                "objective_cost": cost[loop].mean().item(),
                "ece_15_bin": _ece(probabilities[loop], correct[loop]),
                "prediction_change_from_previous": (
                    None
                    if loop == 0
                    else predictions[loop]
                    .ne(predictions[loop - 1])
                    .float()
                    .mean()
                    .item()
                ),
            }
        )

    return {
        "sample_count": count,
        "incorrect_prediction_cost": incorrect_cost,
        "best_fixed_loop_by_objective": best_fixed + 1,
        "per_loop": per_loop,
        "policies": policies,
        "target_histogram": torch.bincount(labels, minlength=classes).tolist(),
        "shuffled_label_oracle_control": {
            "repeats": len(shuffled_accuracy_gains),
            "seed": 1729,
            "permutation_scope": "full_split",
            "accuracy_gain_over_best_fixed_mean": torch.tensor(shuffled_accuracy_gains)
            .mean()
            .item(),
            "accuracy_gain_over_best_fixed_std": torch.tensor(shuffled_accuracy_gains)
            .std(unbiased=False)
            .item(),
            "objective_cost_reduction_over_best_fixed_mean": torch.tensor(
                shuffled_cost_reductions
            )
            .mean()
            .item(),
            "objective_cost_reduction_over_best_fixed_std": torch.tensor(
                shuffled_cost_reductions
            )
            .std(unbiased=False)
            .item(),
            "true_accuracy_gain_over_best_fixed": oracle_accuracy - best_fixed_accuracy,
            "true_objective_cost_reduction_over_best_fixed": denominator,
        },
        "notes": {
            "teacher_causality": "noncausal: bidirectional GRU consumes all loop states",
            "oracle": "label-aware upper bound; not a deployable policy",
            "confidence_policies": "causal earliest threshold crossing; thresholds are descriptive, not tuned",
        },
    }


def _markdown_report(results: dict[str, Any]) -> str:
    lines = [
        "# Backbone and teacher evaluation",
        "",
        "The teacher is a noncausal full-trajectory selector, not an online early-exit policy.",
        "All reported teacher depths therefore describe selection, not realized compute savings.",
        "",
    ]
    for epoch_result in results["epochs"]:
        lines.extend([f"## Backbone epoch {epoch_result['epoch']}", ""])
        for split, analysis in epoch_result["splits"].items():
            lines.extend(
                [
                    f"### {split} ({analysis['sample_count']} examples)",
                    "",
                    "| policy | accuracy | CE | objective | mean loop | oracle regret | headroom captured |",
                    "|---|---:|---:|---:|---:|---:|---:|",
                ]
            )
            selected_names = {
                f"fixed_loop_{analysis['best_fixed_loop_by_objective']}",
                "fixed_loop_6",
                "per_example_oracle",
                "teacher_accuracy_selector",
                "teacher_soft_loop_selector",
                "frozen_halting_head_argmax",
                "causal_confidence_0.70",
                "causal_confidence_0.80",
                "causal_confidence_0.90",
            }
            for policy in analysis["policies"]:
                if policy["name"] not in selected_names:
                    continue
                lines.append(
                    f"| {policy['name']} | {policy['accuracy']:.4f} | "
                    f"{policy['cross_entropy']:.4f} | {policy['objective_cost']:.4f} | "
                    f"{policy['mean_loop']:.2f} | {policy['cost_regret_to_oracle']:.4f} | "
                    f"{policy['headroom_captured_vs_best_fixed']:.3f} |"
                )
            accuracy_teacher = next(
                (
                    policy
                    for policy in analysis["policies"]
                    if policy["name"] == "teacher_accuracy_selector"
                ),
                None,
            )
            if accuracy_teacher is not None:
                guidance = accuracy_teacher["guidance_distribution"]
                distribution = ", ".join(
                    f"{weight:.3f}" for weight in guidance["mean_distribution"]
                )
                lines.extend(
                    [
                        "",
                        "Controller signal: Brier improvement over a per-loop "
                        f"constant predictor = {accuracy_teacher['brier_improvement_over_constant']:.3f}; "
                        f"processed effective depth count = {guidance['mean_effective_depth_count']:.2f}; "
                        f"mean maximum weight = {guidance['mean_max_weight']:.3f}; "
                        f"mean loop weights = [{distribution}].",
                    ]
                )
            control = analysis["shuffled_label_oracle_control"]
            lines.extend(
                [
                    "",
                    "Oracle selection-bias control: true accuracy gain over the best "
                    f"fixed loop = {control['true_accuracy_gain_over_best_fixed']:.4f}; "
                    "shuffled-label gain = "
                    f"{control['accuracy_gain_over_best_fixed_mean']:.4f} "
                    f"± {control['accuracy_gain_over_best_fixed_std']:.4f}. True "
                    "objective reduction = "
                    f"{control['true_objective_cost_reduction_over_best_fixed']:.4f}; "
                    "shuffled-label reduction = "
                    f"{control['objective_cost_reduction_over_best_fixed_mean']:.4f} "
                    f"± {control['objective_cost_reduction_over_best_fixed_std']:.4f}.",
                    "",
                ]
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("conf"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--backbone-checkpoint",
        type=Path,
        help="Explicit backbone checkpoint (must be paired with --teacher-checkpoint)",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Explicit controller checkpoint (must be paired with --backbone-checkpoint)",
    )
    parser.add_argument(
        "--pair-epoch",
        type=int,
        default=0,
        help="Display label for an explicit checkpoint pair",
    )
    parser.add_argument(
        "--data-file",
        action="append",
        type=Path,
        default=None,
        help="Repeat for multiple splits (default: matched and mismatched validation)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-examples", type=int)
    parser.add_argument(
        "--epoch",
        action="append",
        type=int,
        help="Evaluate only this backbone epoch (repeatable; default: all pairs)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--result-dir", type=Path, default=Path("outputs/controller_evaluation")
    )
    args = parser.parse_args()

    data_files = args.data_file or [
        Path("data/mnli_matched.tsv"),
        Path("data/mnli_mismatched.tsv"),
    ]
    if (args.backbone_checkpoint is None) != (args.teacher_checkpoint is None):
        parser.error(
            "--backbone-checkpoint and --teacher-checkpoint must be supplied together"
        )
    if args.backbone_checkpoint is not None:
        pairs = [
            CheckpointPair(
                args.pair_epoch,
                args.backbone_checkpoint,
                args.teacher_checkpoint,
            )
        ]
    else:
        pairs = _checkpoint_pairs(args.output_root)
        if args.epoch:
            selected_epochs = set(args.epoch)
            pairs = [pair for pair in pairs if pair.epoch in selected_epochs]
    if not pairs:
        raise FileNotFoundError("No paired backbone/controller checkpoints found")

    options = load_options(str(args.config_dir), [])
    device = torch.device(args.device)
    tokenizer = MNLITokenizer(options.tokenizer)
    tokenizer_info = tokenizer.get_tokenizer_info()
    teacher_options = ControllerTeacherOptions(
        model_dimension=options.model.hidden_state_dimension,
        ffn_dimension=options.model.transformer.gelu_ffn_dimension,
        heads_count=options.model.transformer.attention_heads_count,
        dropout=options.model.dropout.controller,
    )
    result: dict[str, Any] = {
        "format": "backbone-controller-evaluation-v1",
        "device": str(device),
        "batch_size": args.batch_size,
        "max_examples": args.max_examples,
        "epochs": [],
    }

    for pair in pairs:
        print(f"Evaluating backbone epoch {pair.epoch}", flush=True)
        backbone = LoopedMNLI(
            options.model, options.tokenizer.sequence, tokenizer_info
        ).to(device)
        teacher = ControllerTeacher(teacher_options, options.tokenizer.sequence).to(
            device
        )
        _load_model(backbone, pair.backbone, device)
        _load_model(teacher, pair.teacher, device)
        epoch_result: dict[str, Any] = {
            "epoch": pair.epoch,
            "backbone_checkpoint": str(pair.backbone.resolve()),
            "teacher_checkpoint": str(pair.teacher.resolve()),
            "teacher_loss_type": _teacher_loss_type(pair.teacher),
            "teacher_loss_options": _teacher_loss_options(pair.teacher),
            "splits": {},
        }
        for data_file in data_files:
            outputs = collect_outputs(
                backbone,
                teacher,
                tokenizer,
                data_file,
                args.batch_size,
                device,
                args.max_examples,
            )
            epoch_result["splits"][data_file.stem] = analyse(
                outputs,
                teacher_loss_options=epoch_result["teacher_loss_options"],
            )
        result["epochs"].append(epoch_result)
        del backbone, teacher
        if device.type == "cuda":
            torch.cuda.empty_cache()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.result_dir / "analysis.json"
    report_path = args.result_dir / "report.md"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    report_path.write_text(_markdown_report(result), encoding="utf-8")
    print(f"Wrote {json_path} and {report_path}", flush=True)


if __name__ == "__main__":
    main()
