from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor
from torch.nn import functional

from looped_transformer.abstractions.data import MNLIBatch
from looped_transformer.abstractions.mnli import get_logic_relation
from looped_transformer.abstractions.model import ModelForwardOutput, TypedModule
from looped_transformer.config.base import (
    ExclusiveProbability,
    NonNegativeInt,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.config.runtime import DeviceOptions
from looped_transformer.core.evaluation import EvaluationPolicy
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.modules.config import LoopOptions


class LoopDiagnosticsOptions(OptionsModel):
    label_smoothing: ExclusiveProbability = 0.05
    shuffle_oracle_repeats: PositiveInt = 20
    shuffle_seed: NonNegativeInt = 0


@dataclass(frozen=True, slots=True)
class BatchDiagnostics:
    batch_index: int
    sample_count: int
    primary: dict[str, float]
    final_loop: dict[str, float]
    mean_across_loops: dict[str, float]
    best_fixed_loop: dict[str, float | int]
    per_sample_oracle: dict[str, float]
    shuffled_label_oracle: dict[str, Any]
    oracle_lift_over_shuffled: dict[str, float]
    adaptive_headroom: dict[str, float]
    per_loop: dict[str, list[float] | list[int]]
    prediction_trajectory: dict[str, list[float]]
    class_histograms: dict[str, list[int] | list[list[int]]]


@dataclass(slots=True)
class _MetricTotals:
    raw_cross_entropy: float = 0.0
    smooth_cross_entropy: float = 0.0
    correct: float = 0.0

    def add(
        self,
        raw_cross_entropy: Tensor,
        smooth_cross_entropy: Tensor,
        correct: Tensor,
    ) -> None:
        self.raw_cross_entropy += raw_cross_entropy.sum().item()
        self.smooth_cross_entropy += smooth_cross_entropy.sum().item()
        self.correct += correct.sum().item()

    def means(self, sample_count: int) -> dict[str, float]:
        return {
            "raw_cross_entropy": self.raw_cross_entropy / sample_count,
            "smooth_cross_entropy": self.smooth_cross_entropy / sample_count,
            "accuracy": self.correct / sample_count,
        }


class LoopDiagnosticsPolicy(
    EvaluationPolicy[MNLIBatch, Tensor, Tensor, ModelForwardOutput]
):
    _target_histogram: Tensor | None
    _final_prediction_histogram: Tensor | None
    _per_loop_prediction_histogram: Tensor | None

    def __init__(
        self,
        loop_options: LoopOptions,
        device_options: DeviceOptions,
        options: LoopDiagnosticsOptions,
        logger: StructuralLogger,
    ) -> None:
        self._max_iterations = loop_options.max_iterations
        self._device = device_options.device
        self._label_smoothing = options.label_smoothing
        self._shuffle_oracle_repeats = options.shuffle_oracle_repeats
        self._shuffle_seed = options.shuffle_seed
        self._logger = logger
        self._initialize_state()

    def _initialize_state(self) -> None:
        self._batch_count = 0
        self._sample_count = 0
        self._primary_totals = _MetricTotals()
        self._final_loop_totals = _MetricTotals()
        self._mean_loop_totals = _MetricTotals()
        self._oracle_totals = _MetricTotals()

        self._per_loop_raw_cross_entropy = torch.zeros(
            self._max_iterations, device=self._device
        )
        self._per_loop_smooth_cross_entropy = torch.zeros(
            self._max_iterations, device=self._device
        )
        self._per_loop_correct = torch.zeros(self._max_iterations, device=self._device)
        self._oracle_best_loop_histogram = torch.zeros(
            self._max_iterations, dtype=torch.long, device=self._device
        )

        repeat_count = self._shuffle_oracle_repeats
        self._shuffled_raw_cross_entropy = torch.zeros(
            repeat_count, device=self._device
        )
        self._shuffled_smooth_cross_entropy = torch.zeros(
            repeat_count, device=self._device
        )
        self._shuffled_correct = torch.zeros(repeat_count, device=self._device)
        self._shuffle_generator = torch.Generator().manual_seed(self._shuffle_seed)

        self._final_to_oracle_raw_gain = 0.0
        self._oracle_raw_improvements_over_final = 0
        self._oracle_rescues = 0
        self._oracle_harms = 0
        self._final_errors = 0

        self._adjacent_prediction_changes = torch.zeros(
            max(0, self._max_iterations - 1), device=self._device
        )
        self._final_prediction_agreements = torch.zeros(
            self._max_iterations, device=self._device
        )
        self._target_histogram: Tensor | None = None
        self._final_prediction_histogram: Tensor | None = None
        self._per_loop_prediction_histogram: Tensor | None = None

    @override
    def reset(self, model: TypedModule[Tensor, Tensor, ModelForwardOutput]) -> None:
        self._initialize_state()
        self._logger.start()

    @override
    def close(self) -> None:
        pass

    @staticmethod
    def _cross_entropy_by_loop(
        timed_logits: Tensor,
        labels: Tensor,
        label_smoothing: float = 0.0,
    ) -> Tensor:
        loop_count, sample_count, class_count = timed_logits.shape
        flat_labels = labels.unsqueeze(0).expand(loop_count, -1).reshape(-1)
        return functional.cross_entropy(
            timed_logits.reshape(loop_count * sample_count, class_count),
            flat_labels,
            reduction="none",
            label_smoothing=label_smoothing,
        ).reshape(loop_count, sample_count)

    @staticmethod
    def _round_values(values: Tensor) -> list[float]:
        return [round(float(value), 6) for value in values.detach().cpu().tolist()]

    @staticmethod
    def _metrics(
        raw_cross_entropy: Tensor,
        smooth_cross_entropy: Tensor,
        correct: Tensor,
    ) -> dict[str, float]:
        return {
            "raw_cross_entropy": round(raw_cross_entropy.mean().item(), 6),
            "smooth_cross_entropy": round(smooth_cross_entropy.mean().item(), 6),
            "accuracy": round(correct.float().mean().item(), 6),
        }

    def _validate_output(self, timed_logits: Tensor) -> None:
        if timed_logits.ndim != 3:
            raise ValueError(
                "Expected timed_logits with shape (loops, samples, classes), "
                f"got {tuple(timed_logits.shape)}"
            )
        if timed_logits.size(0) != self._max_iterations:
            raise ValueError(
                f"Expected {self._max_iterations} loops, got {timed_logits.size(0)}"
            )
        if timed_logits.size(2) <= 1:
            raise ValueError("timed_logits must contain at least two classes")
        if not torch.isfinite(timed_logits).all():
            raise ValueError("timed_logits contains non-finite values")

    def _shuffled_oracle_metrics(
        self,
        timed_logits: Tensor,
        labels: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        sample_count = labels.numel()
        batch_indices = torch.arange(sample_count, device=timed_logits.device)
        raw_values = []
        smooth_values = []
        correct_values = []

        for _ in range(self._shuffle_oracle_repeats):
            permutation = torch.randperm(
                sample_count, generator=self._shuffle_generator
            ).to(labels.device)
            shuffled_labels = labels[permutation]
            raw_by_loop = self._cross_entropy_by_loop(timed_logits, shuffled_labels)
            smooth_by_loop = self._cross_entropy_by_loop(
                timed_logits, shuffled_labels, self._label_smoothing
            )
            best_loops = raw_by_loop.argmin(dim=0)
            predictions = timed_logits[best_loops, batch_indices].argmax(dim=-1)

            raw_values.append(raw_by_loop[best_loops, batch_indices])
            smooth_values.append(smooth_by_loop[best_loops, batch_indices])
            correct_values.append((predictions == shuffled_labels).float())

        return (
            torch.stack(raw_values),
            torch.stack(smooth_values),
            torch.stack(correct_values),
        )

    @override
    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        timed_logits = output["timed_logits"].float()
        self._validate_output(timed_logits)
        loop_count, sample_count, class_count = timed_logits.shape

        labels = torch.tensor(
            [get_logic_relation(label).value for label in batch.labels],
            dtype=torch.long,
            device=timed_logits.device,
        )
        if labels.numel() != sample_count:
            raise ValueError(
                "Batch size mismatch: "
                f"received {labels.numel()} labels for {sample_count} outputs"
            )

        raw_by_loop = self._cross_entropy_by_loop(timed_logits, labels)
        smooth_by_loop = self._cross_entropy_by_loop(
            timed_logits, labels, self._label_smoothing
        )
        predictions_by_loop = timed_logits.argmax(dim=-1)
        correct_by_loop = predictions_by_loop == labels.unsqueeze(0)

        final_raw = raw_by_loop[-1]
        final_smooth = smooth_by_loop[-1]
        final_correct = correct_by_loop[-1]
        mean_raw = raw_by_loop.mean(dim=0)
        mean_smooth = smooth_by_loop.mean(dim=0)
        mean_correct = correct_by_loop.float().mean(dim=0)

        batch_indices = torch.arange(sample_count, device=timed_logits.device)
        oracle_best_loops = raw_by_loop.argmin(dim=0)
        oracle_raw = raw_by_loop[oracle_best_loops, batch_indices]
        oracle_smooth = smooth_by_loop[oracle_best_loops, batch_indices]
        oracle_predictions = timed_logits[oracle_best_loops, batch_indices].argmax(
            dim=-1
        )
        oracle_correct = oracle_predictions == labels

        per_loop_raw = raw_by_loop.sum(dim=1)
        per_loop_smooth = smooth_by_loop.sum(dim=1)
        per_loop_correct = correct_by_loop.sum(dim=1)
        best_fixed_loop = int(per_loop_raw.argmin().item())
        best_fixed_raw = raw_by_loop[best_fixed_loop]
        best_fixed_smooth = smooth_by_loop[best_fixed_loop]
        best_fixed_correct = correct_by_loop[best_fixed_loop]

        shuffled_raw, shuffled_smooth, shuffled_correct = self._shuffled_oracle_metrics(
            timed_logits, labels
        )
        shuffled_metrics = {
            "raw_cross_entropy": shuffled_raw.mean().item(),
            "smooth_cross_entropy": shuffled_smooth.mean().item(),
            "accuracy": shuffled_correct.mean().item(),
            "accuracy_std_across_repeats": shuffled_correct.mean(dim=1)
            .std(unbiased=False)
            .item(),
        }

        primary_raw = final_raw
        primary_smooth = final_smooth
        primary_correct = final_correct.float()

        self._primary_totals.add(primary_raw, primary_smooth, primary_correct)
        self._final_loop_totals.add(final_raw, final_smooth, final_correct)
        self._mean_loop_totals.add(mean_raw, mean_smooth, mean_correct)
        self._oracle_totals.add(oracle_raw, oracle_smooth, oracle_correct)

        self._per_loop_raw_cross_entropy += per_loop_raw
        self._per_loop_smooth_cross_entropy += per_loop_smooth
        self._per_loop_correct += per_loop_correct
        oracle_histogram = torch.bincount(oracle_best_loops, minlength=loop_count)
        self._oracle_best_loop_histogram += oracle_histogram
        self._shuffled_raw_cross_entropy += shuffled_raw.sum(dim=1)
        self._shuffled_smooth_cross_entropy += shuffled_smooth.sum(dim=1)
        self._shuffled_correct += shuffled_correct.sum(dim=1)

        final_to_oracle_gain = final_raw - oracle_raw
        oracle_rescues = (~final_correct) & oracle_correct
        oracle_harms = final_correct & (~oracle_correct)
        self._final_to_oracle_raw_gain += final_to_oracle_gain.sum().item()
        self._oracle_raw_improvements_over_final += int(
            (final_to_oracle_gain > 0).sum().item()
        )
        self._oracle_rescues += int(oracle_rescues.sum().item())
        self._oracle_harms += int(oracle_harms.sum().item())
        self._final_errors += int((~final_correct).sum().item())

        if loop_count > 1:
            self._adjacent_prediction_changes += (
                predictions_by_loop[1:] != predictions_by_loop[:-1]
            ).sum(dim=1)
        self._final_prediction_agreements += (
            predictions_by_loop == predictions_by_loop[-1:]
        ).sum(dim=1)

        target_histogram = torch.bincount(labels, minlength=class_count)
        final_histogram = torch.bincount(predictions_by_loop[-1], minlength=class_count)
        per_loop_histogram = torch.stack(
            [
                torch.bincount(predictions, minlength=class_count)
                for predictions in predictions_by_loop
            ]
        )
        # noinspection attribute-outside-init
        if self._target_histogram is None:
            self._target_histogram = torch.zeros_like(target_histogram)
            self._final_prediction_histogram = torch.zeros_like(final_histogram)
            self._per_loop_prediction_histogram = torch.zeros_like(per_loop_histogram)
        assert self._final_prediction_histogram is not None
        assert self._per_loop_prediction_histogram is not None
        self._target_histogram += target_histogram
        self._final_prediction_histogram += final_histogram
        self._per_loop_prediction_histogram += per_loop_histogram

        self._sample_count += sample_count
        self._batch_count += 1

        batch_diagnostics = BatchDiagnostics(
            batch_index=self._batch_count,
            sample_count=sample_count,
            primary=self._metrics(primary_raw, primary_smooth, primary_correct),
            final_loop=self._metrics(final_raw, final_smooth, final_correct),
            mean_across_loops=self._metrics(mean_raw, mean_smooth, mean_correct),
            best_fixed_loop={
                "loop": best_fixed_loop + 1,
                **self._metrics(best_fixed_raw, best_fixed_smooth, best_fixed_correct),
            },
            per_sample_oracle=self._metrics(oracle_raw, oracle_smooth, oracle_correct),
            shuffled_label_oracle={
                key: round(value, 6) for key, value in shuffled_metrics.items()
            },
            oracle_lift_over_shuffled={
                "raw_cross_entropy_reduction": round(
                    shuffled_metrics["raw_cross_entropy"] - oracle_raw.mean().item(),
                    6,
                ),
                "accuracy_gain": round(
                    oracle_correct.float().mean().item() - shuffled_metrics["accuracy"],
                    6,
                ),
            },
            adaptive_headroom={
                "raw_cross_entropy_reduction_over_final": round(
                    final_to_oracle_gain.mean().item(), 6
                ),
                "raw_cross_entropy_improvement_rate_over_final": round(
                    (final_to_oracle_gain > 0).float().mean().item(), 6
                ),
                "raw_cross_entropy_reduction_over_best_fixed": round(
                    (best_fixed_raw - oracle_raw).mean().item(), 6
                ),
                "accuracy_gain_over_final": round(
                    oracle_correct.float().mean().item()
                    - final_correct.float().mean().item(),
                    6,
                ),
                "accuracy_gain_over_best_fixed": round(
                    oracle_correct.float().mean().item()
                    - best_fixed_correct.float().mean().item(),
                    6,
                ),
                "final_error_rescue_rate": round(
                    oracle_rescues.float().mean().item(), 6
                ),
                "conditional_final_error_rescue_rate": round(
                    oracle_rescues.sum().item() / max(1, (~final_correct).sum().item()),
                    6,
                ),
                "final_correct_harm_rate": round(oracle_harms.float().mean().item(), 6),
            },
            per_loop={
                "loop": list(range(1, loop_count + 1)),
                "raw_cross_entropy": self._round_values(raw_by_loop.mean(dim=1)),
                "smooth_cross_entropy": self._round_values(smooth_by_loop.mean(dim=1)),
                "accuracy": self._round_values(correct_by_loop.float().mean(dim=1)),
                "oracle_best_loop_histogram": oracle_histogram.cpu().tolist(),
            },
            prediction_trajectory={
                "adjacent_change_rate": self._round_values(
                    (predictions_by_loop[1:] != predictions_by_loop[:-1])
                    .float()
                    .mean(dim=1)
                ),
                "final_agreement_rate": self._round_values(
                    (predictions_by_loop == predictions_by_loop[-1:])
                    .float()
                    .mean(dim=1)
                ),
            },
            class_histograms={
                "target": target_histogram.cpu().tolist(),
                "final_prediction": final_histogram.cpu().tolist(),
                "prediction_by_loop": per_loop_histogram.cpu().tolist(),
            },
        )
        self._logger.log_batch_evaluation(batch_diagnostics)

    def collect_diagnostics(self) -> dict[str, Any]:
        self._require_samples()
        sample_count = self._sample_count
        per_loop_raw = self._per_loop_raw_cross_entropy / sample_count
        per_loop_smooth = self._per_loop_smooth_cross_entropy / sample_count
        per_loop_accuracy = self._per_loop_correct / sample_count
        best_fixed_loop = int(per_loop_raw.argmin().item())

        oracle_metrics = self._oracle_totals.means(sample_count)
        shuffled_raw_by_repeat = self._shuffled_raw_cross_entropy / sample_count
        shuffled_smooth_by_repeat = self._shuffled_smooth_cross_entropy / sample_count
        shuffled_accuracy_by_repeat = self._shuffled_correct / sample_count
        shuffled_metrics = {
            "repeats": self._shuffle_oracle_repeats,
            "seed": self._shuffle_seed,
            "permutation_scope": "within_batch",
            "raw_cross_entropy_mean": shuffled_raw_by_repeat.mean().item(),
            "raw_cross_entropy_std": shuffled_raw_by_repeat.std(unbiased=False).item(),
            "smooth_cross_entropy_mean": shuffled_smooth_by_repeat.mean().item(),
            "smooth_cross_entropy_std": shuffled_smooth_by_repeat.std(
                unbiased=False
            ).item(),
            "accuracy_mean": shuffled_accuracy_by_repeat.mean().item(),
            "accuracy_std": shuffled_accuracy_by_repeat.std(unbiased=False).item(),
            "accuracy_by_repeat": shuffled_accuracy_by_repeat.cpu().tolist(),
        }

        best_fixed_metrics = {
            "selection_criterion": "minimum aggregate raw cross-entropy",
            "loop": best_fixed_loop + 1,
            "raw_cross_entropy": per_loop_raw[best_fixed_loop].item(),
            "smooth_cross_entropy": per_loop_smooth[best_fixed_loop].item(),
            "accuracy": per_loop_accuracy[best_fixed_loop].item(),
        }
        final_metrics = self._final_loop_totals.means(sample_count)

        assert self._target_histogram is not None
        assert self._final_prediction_histogram is not None
        assert self._per_loop_prediction_histogram is not None

        return {
            "batch_count": self._batch_count,
            "sample_count": sample_count,
            "primary": self._primary_totals.means(sample_count),
            "final_loop": final_metrics,
            "mean_across_loops": self._mean_loop_totals.means(sample_count),
            "best_fixed_loop": best_fixed_metrics,
            "per_sample_oracle": {
                **oracle_metrics,
                "selection_criterion": "minimum per-sample raw cross-entropy",
            },
            "shuffled_label_oracle": shuffled_metrics,
            "oracle_lift_over_shuffled": {
                "raw_cross_entropy_reduction": (
                    shuffled_metrics["raw_cross_entropy_mean"]
                    - oracle_metrics["raw_cross_entropy"]
                ),
                "accuracy_gain": (
                    oracle_metrics["accuracy"] - shuffled_metrics["accuracy_mean"]
                ),
            },
            "adaptive_headroom": {
                "raw_cross_entropy_reduction_over_final": (
                    self._final_to_oracle_raw_gain / sample_count
                ),
                "raw_cross_entropy_improvement_rate_over_final": (
                    self._oracle_raw_improvements_over_final / sample_count
                ),
                "raw_cross_entropy_reduction_over_best_fixed": (
                    best_fixed_metrics["raw_cross_entropy"]
                    - oracle_metrics["raw_cross_entropy"]
                ),
                "accuracy_gain_over_final": (
                    oracle_metrics["accuracy"] - final_metrics["accuracy"]
                ),
                "accuracy_gain_over_best_fixed": (
                    oracle_metrics["accuracy"] - best_fixed_metrics["accuracy"]
                ),
                "final_error_rescue_rate": self._oracle_rescues / sample_count,
                "conditional_final_error_rescue_rate": (
                    self._oracle_rescues / max(1, self._final_errors)
                ),
                "final_correct_harm_rate": self._oracle_harms / sample_count,
            },
            "per_loop": {
                "loop": list(range(1, self._max_iterations + 1)),
                "raw_cross_entropy": self._round_values(per_loop_raw),
                "smooth_cross_entropy": self._round_values(per_loop_smooth),
                "accuracy": self._round_values(per_loop_accuracy),
                "oracle_best_loop_histogram": (
                    self._oracle_best_loop_histogram.cpu().tolist()
                ),
            },
            "prediction_trajectory": {
                "adjacent_change_rate": self._round_values(
                    self._adjacent_prediction_changes / sample_count
                ),
                "final_agreement_rate": self._round_values(
                    self._final_prediction_agreements / sample_count
                ),
            },
            "class_histograms": {
                "target": self._target_histogram.cpu().tolist(),
                "final_prediction": self._final_prediction_histogram.cpu().tolist(),
                "prediction_by_loop": (
                    self._per_loop_prediction_histogram.cpu().tolist()
                ),
            },
        }

    def _require_samples(self) -> None:
        if self._sample_count == 0:
            raise ValueError("Evaluation data contains no recognized labels")
