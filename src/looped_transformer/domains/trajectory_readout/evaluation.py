from dataclasses import dataclass
from typing import Any, override

import torch
from torch import Tensor
from torch.nn import functional

from looped_transformer.core.abstractions.typed_module import TypedBranchedModule
from looped_transformer.core.evaluation import EvaluationPolicy
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.domains.clutrr.types import CLUTRR_RELATIONS, relation_index
from looped_transformer.domains.trajectory_readout.batch import TrajectoryBatch


@dataclass(frozen=True, slots=True)
class ReadoutBatchDiagnostics:
    batch_index: int
    sample_count: int
    readout: dict[str, float]
    backbone_final: dict[str, float]
    lift_over_backbone: dict[str, float]
    by_reasoning_depth: dict[str, dict[str, Any]]


@dataclass(slots=True)
class _MetricTotals:
    raw_cross_entropy: float = 0.0
    smooth_cross_entropy: float = 0.0
    correct: float = 0.0
    true_label_probability: float = 0.0
    prediction_confidence: float = 0.0

    def add(self, metrics: "_BatchMetrics") -> None:
        self.raw_cross_entropy += metrics.raw_cross_entropy.sum().item()
        self.smooth_cross_entropy += metrics.smooth_cross_entropy.sum().item()
        self.correct += metrics.correct.sum().item()
        self.true_label_probability += metrics.true_label_probability.sum().item()
        self.prediction_confidence += metrics.prediction_confidence.sum().item()

    def means(self, sample_count: int) -> dict[str, float]:
        return {
            "raw_cross_entropy": self.raw_cross_entropy / sample_count,
            "smooth_cross_entropy": self.smooth_cross_entropy / sample_count,
            "accuracy": self.correct / sample_count,
            "true_label_probability": self.true_label_probability / sample_count,
            "prediction_confidence": self.prediction_confidence / sample_count,
        }


@dataclass(frozen=True, slots=True)
class _BatchMetrics:
    raw_cross_entropy: Tensor
    smooth_cross_entropy: Tensor
    correct: Tensor
    true_label_probability: Tensor
    prediction_confidence: Tensor
    predictions: Tensor

    def means(self) -> dict[str, float]:
        return {
            "raw_cross_entropy": self.raw_cross_entropy.mean().item(),
            "smooth_cross_entropy": self.smooth_cross_entropy.mean().item(),
            "accuracy": self.correct.float().mean().item(),
            "true_label_probability": self.true_label_probability.mean().item(),
            "prediction_confidence": self.prediction_confidence.mean().item(),
        }


class TrajectoryReadoutEvaluationPolicy(
    EvaluationPolicy[TrajectoryBatch, Tensor, Tensor, Tensor]
):
    """Aggregate final readout metrics and compare them with the frozen backbone."""

    def __init__(self, label_smoothing: float, logger: StructuralLogger) -> None:
        self._label_smoothing = label_smoothing
        self._logger = logger
        self._initialize_state()

    def _initialize_state(self) -> None:
        self._batch_count = 0
        self._sample_count = 0
        self._readout_totals = _MetricTotals()
        self._backbone_totals = _MetricTotals()
        self._target_histogram = torch.zeros(len(CLUTRR_RELATIONS), dtype=torch.long)
        self._readout_histogram = torch.zeros_like(self._target_histogram)
        self._backbone_histogram = torch.zeros_like(self._target_histogram)
        self._depth_counts: dict[int, int] = {}
        self._depth_readout: dict[int, _MetricTotals] = {}
        self._depth_backbone: dict[int, _MetricTotals] = {}
        self._depth_targets: dict[int, Tensor] = {}

    @override
    def reset(self, model: TypedBranchedModule[Tensor, Tensor, Tensor]) -> None:
        del model
        self._initialize_state()
        self._logger.start()

    @override
    def attend_batch(self, batch: TrajectoryBatch, output: Tensor) -> None:
        labels = torch.tensor(
            [relation_index(label) for label in batch.source.labels],
            device=output.device,
            dtype=torch.long,
        )
        readout = self._metrics(output, labels)
        backbone = self._metrics(batch.final_logits.to(output.device), labels)
        sample_count = labels.numel()

        self._readout_totals.add(readout)
        self._backbone_totals.add(backbone)
        self._target_histogram += torch.bincount(
            labels.cpu(), minlength=len(CLUTRR_RELATIONS)
        )
        self._readout_histogram += torch.bincount(
            readout.predictions.cpu(), minlength=len(CLUTRR_RELATIONS)
        )
        self._backbone_histogram += torch.bincount(
            backbone.predictions.cpu(), minlength=len(CLUTRR_RELATIONS)
        )

        depth_diagnostics: dict[str, dict[str, Any]] = {}
        depths = torch.tensor(batch.source.depths, device=output.device)
        for depth in sorted(set(batch.source.depths)):
            selected = depths == depth
            count = int(selected.sum().item())
            readout_depth = self._select(readout, selected)
            backbone_depth = self._select(backbone, selected)
            self._depth_counts[depth] = self._depth_counts.get(depth, 0) + count
            self._depth_readout.setdefault(depth, _MetricTotals()).add(readout_depth)
            self._depth_backbone.setdefault(depth, _MetricTotals()).add(backbone_depth)
            depth_targets = self._depth_targets.setdefault(
                depth, torch.zeros(len(CLUTRR_RELATIONS), dtype=torch.long)
            )
            depth_targets += torch.bincount(
                labels[selected].cpu(), minlength=len(CLUTRR_RELATIONS)
            )
            depth_diagnostics[str(depth)] = self._comparison(
                readout_depth.means(), backbone_depth.means(), count
            )

        self._sample_count += sample_count
        self._batch_count += 1
        readout_means = readout.means()
        backbone_means = backbone.means()
        self._logger.log_batch_evaluation(
            ReadoutBatchDiagnostics(
                batch_index=self._batch_count,
                sample_count=sample_count,
                readout=readout_means,
                backbone_final=backbone_means,
                lift_over_backbone=self._lift(readout_means, backbone_means),
                by_reasoning_depth=depth_diagnostics,
            )
        )

    @override
    def close(self) -> None:
        pass

    def collect_diagnostics(self) -> dict[str, Any]:
        if self._sample_count == 0:
            raise ValueError("Evaluation data contains no samples")
        readout = self._readout_totals.means(self._sample_count)
        backbone = self._backbone_totals.means(self._sample_count)
        by_depth = {}
        for depth, count in sorted(self._depth_counts.items()):
            by_depth[str(depth)] = {
                **self._comparison(
                    self._depth_readout[depth].means(count),
                    self._depth_backbone[depth].means(count),
                    count,
                ),
                "target_histogram": self._depth_targets[depth].tolist(),
            }
        return {
            "batch_count": self._batch_count,
            "sample_count": self._sample_count,
            "readout": readout,
            "backbone_final": backbone,
            "lift_over_backbone": self._lift(readout, backbone),
            "class_histograms": {
                "labels": list(CLUTRR_RELATIONS),
                "target": self._target_histogram.tolist(),
                "readout_prediction": self._readout_histogram.tolist(),
                "backbone_final_prediction": self._backbone_histogram.tolist(),
            },
            "by_reasoning_depth": by_depth,
        }

    def _metrics(self, logits: Tensor, labels: Tensor) -> _BatchMetrics:
        float_logits = logits.float()
        probabilities = float_logits.softmax(dim=-1)
        confidence, predictions = probabilities.max(dim=-1)
        return _BatchMetrics(
            raw_cross_entropy=functional.cross_entropy(
                float_logits, labels, reduction="none"
            ),
            smooth_cross_entropy=functional.cross_entropy(
                float_logits,
                labels,
                reduction="none",
                label_smoothing=self._label_smoothing,
            ),
            correct=predictions == labels,
            true_label_probability=probabilities.gather(1, labels.unsqueeze(1)).squeeze(
                1
            ),
            prediction_confidence=confidence,
            predictions=predictions,
        )

    @staticmethod
    def _select(metrics: _BatchMetrics, selected: Tensor) -> _BatchMetrics:
        return _BatchMetrics(
            raw_cross_entropy=metrics.raw_cross_entropy[selected],
            smooth_cross_entropy=metrics.smooth_cross_entropy[selected],
            correct=metrics.correct[selected],
            true_label_probability=metrics.true_label_probability[selected],
            prediction_confidence=metrics.prediction_confidence[selected],
            predictions=metrics.predictions[selected],
        )

    @staticmethod
    def _lift(
        readout: dict[str, float], backbone: dict[str, float]
    ) -> dict[str, float]:
        return {
            "raw_cross_entropy_reduction": (
                backbone["raw_cross_entropy"] - readout["raw_cross_entropy"]
            ),
            "smooth_cross_entropy_reduction": (
                backbone["smooth_cross_entropy"] - readout["smooth_cross_entropy"]
            ),
            "accuracy_gain": readout["accuracy"] - backbone["accuracy"],
        }

    @classmethod
    def _comparison(
        cls,
        readout: dict[str, float],
        backbone: dict[str, float],
        sample_count: int,
    ) -> dict[str, Any]:
        return {
            "sample_count": sample_count,
            "readout": readout,
            "backbone_final": backbone,
            "lift_over_backbone": cls._lift(readout, backbone),
        }
