"""Post-hook geometry diagnostics; no changes to model execution or training."""

import hashlib
import math
from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor

from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.mnli import get_logic_relation


class WarmupAnalyzer:
    """Compare observed loop trajectories, without retaining activation tensors.

    get_report() returns structured values without writing files.
    Persistence and presentation belong to the analysis command and document tools.
    Use identical examples/order and evaluation settings for meaningful comparisons.
    Only observed loops are measured: existing early halting is not overridden.
    """

    def __init__(
        self,
        model: LoopedMNLI,
        model_options,
        labels: Iterator[MNLIBatch],
        *,
        run_name: str = "warmup",
        max_token_samples: int = 16,
    ):
        self._model = model
        self._labels = labels
        self._max_iterations = model_options.max_iterations
        self._run_name = run_name
        self._max_token_samples = max(0, max_token_samples)
        self._handles: list[Any] = []
        self._sample_offset = 0
        self._current_t = -1
        self._current_labels = None
        self._sample_ids: list[int] = []
        self._sample_keys: dict[int, str] = {}
        self._rows: dict[int, dict[int, dict[str, float | None]]] = defaultdict(dict)
        self._tokens: dict[int, dict[int, dict]] = defaultdict(dict)
        self._previous: dict[str, Tensor] = {}
        self._current_h_curr = None
        self._mask = None
        self._head_basis = None
        self._head_version = None
        self._register_hooks()

    def _register_hooks(self):
        self._handles.extend(
            [
                self._model._loop.register_forward_pre_hook(self._loop_pre_hook),
                self._model._loop.register_forward_hook(self._loop_post_hook),
                self._model._halting_cell.register_forward_hook(self._halting_hook),
                self._model._classifier.register_forward_hook(self._classifier_hook),
            ]
        )

    def remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self._previous.clear()
        self._current_h_curr = None

    @staticmethod
    def _masked_mean(x: Tensor, mask: Tensor) -> Tensor:
        valid = mask.bool()
        count = valid.sum(1)
        mean = x.masked_fill(~valid, 0).sum(1) / count.clamp_min(1)
        return torch.where(count > 0, mean, float("nan"))

    @staticmethod
    def _pool(x: Tensor, mask: Tensor) -> Tensor:
        return x.masked_fill(~mask.bool().unsqueeze(-1), 0).sum(1) / mask.bool().sum(
            1
        ).clamp_min(1).unsqueeze(-1)

    @staticmethod
    def _ratio(numerator: Tensor, denominator: Tensor) -> Tensor:
        # A zero-energy state's fraction/direction is undefined, not zero.
        return torch.where(
            denominator > 1e-12, numerator / denominator.clamp_min(1e-12), float("nan")
        )

    @staticmethod
    def _cos(a: Tensor, b: Tensor) -> Tensor:
        return WarmupAnalyzer._ratio(
            (a * b).sum(-1), a.norm(dim=-1) * b.norm(dim=-1)
        ).clamp(-1, 1)

    def _record(self, t: int, **metrics: Tensor):
        values = {
            key: value.detach().float().cpu().tolist() for key, value in metrics.items()
        }
        for i, sid in enumerate(self._sample_ids):
            row = self._rows[sid].setdefault(t, {})
            for key, value in values.items():
                number = value[i]
                row[key] = (
                    number if number is not None and math.isfinite(number) else None
                )

    def _token_record(self, t: int, mask: Tensor, **metrics: Tensor):
        for i, sid in enumerate(self._sample_ids):
            if sid >= self._max_token_samples:
                continue
            valid = mask[i].bool()
            row = self._tokens[sid].setdefault(t, {})
            row["positions"] = valid.nonzero().flatten().cpu().tolist()
            for key, value in metrics.items():
                row[key] = [
                    v if math.isfinite(v) else None
                    for v in value[i][valid].cpu().tolist()
                ]

    def _trajectory(
        self,
        name: str,
        current: Tensor,
        t: int,
        mask: Tensor | None = None,
        previous: Tensor | None = None,
    ):
        prev = previous if previous is not None else self._previous.get(name)
        norm = current.norm(dim=-1)
        metrics = {f"{name}_norm": norm}
        if prev is not None:
            delta = (current - prev).norm(dim=-1)
            metrics.update(
                {
                    f"{name}_delta": delta,
                    f"{name}_relative_delta": self._ratio(delta, prev.norm(dim=-1)),
                    f"{name}_cos": self._cos(current, prev),
                }
            )
        if mask is not None:
            self._token_record(t, mask, **metrics)
            metrics = {
                key: self._masked_mean(value, mask) for key, value in metrics.items()
            }
        self._record(t, **metrics)
        self._previous[name] = current

    def _loop_pre_hook(self, _, inputs):
        t, h0 = inputs[:2]
        self._current_t = t
        if t == 0:
            batch = next(self._labels)
            labels = [get_logic_relation(label).value for label in batch.labels]
            if len(labels) != h0.shape[0]:
                raise ValueError(
                    "Analyzer labels must exactly match the model batch; use identical data order."
                )
            self._current_labels = torch.tensor(labels, device=h0.device)
            self._sample_ids = list(
                range(self._sample_offset, self._sample_offset + len(labels))
            )
            for sid, premise, hypothesis, label in zip(
                self._sample_ids,
                batch.premises,
                batch.hypotheses,
                batch.labels,
                strict=True,
            ):
                identity = "\0".join((premise, hypothesis, label.strip().lower()))
                self._sample_keys[sid] = hashlib.sha256(identity.encode()).hexdigest()
            self._sample_offset += len(labels)
            self._previous.clear()
            self._current_h_curr = None

    @torch.no_grad()
    def _loop_post_hook(self, _, inputs, output):
        t, _, previous, mask = inputs[:4]
        current = output.detach().float()
        previous = previous.detach().float()
        self._mask = mask.bool()
        self._current_h_curr = current
        self._trajectory("loop", current, t, self._mask, previous)
        self._record(t, valid_tokens=self._mask.sum(1).float())
        # In this model each loop applies cos[0]/sin[0]. Undo that actual
        # rotation to isolate the current loop's pre-rotation transformation.
        cos, sin = inputs[4][0].float(), inputs[5][0].float()
        a, b = current.chunk(2, -1)
        unrotated = torch.cat((a * cos + b * sin, b * cos - a * sin), -1)
        self._record(
            t,
            pre_rotation_delta=self._masked_mean(
                (unrotated - previous).norm(dim=-1), self._mask
            ),
            pre_rotation_cos=self._masked_mean(
                self._cos(unrotated, previous), self._mask
            ),
            rotation_delta=self._masked_mean(
                (current - unrotated).norm(dim=-1), self._mask
            ),
            rotation_norm_error=self._masked_mean(
                (current.norm(dim=-1) - unrotated.norm(dim=-1)).abs(), self._mask
            ),
        )

    @torch.no_grad()
    def _halting_hook(self, _, inputs, __):
        t, aligned, _, mask = inputs
        aligned = aligned.detach().float()
        mask = mask.bool()
        pooled = self._pool(aligned, mask)
        residual = aligned - pooled.unsqueeze(1)
        energy = self._masked_mean(aligned.square().sum(-1), mask)
        residual_energy = self._masked_mean(residual.square().sum(-1), mask)
        metrics = dict(
            pooling_residual_fraction=self._ratio(residual_energy, energy),
            token_spread=residual_energy.sqrt(),
        )
        prev = self._previous.get("aligned")
        if prev is not None:
            movement = aligned - prev
            pooled_movement = self._pool(movement, mask)
            movement_energy = self._masked_mean(movement.square().sum(-1), mask)
            cancelled_energy = self._masked_mean(
                (movement - pooled_movement.unsqueeze(1)).square().sum(-1), mask
            )
            metrics["pooling_cancelled_motion_fraction"] = self._ratio(
                cancelled_energy, movement_energy
            )
        raw = self._current_h_curr
        if raw is not None:
            metrics["h_aligned_delta"] = self._masked_mean(
                (raw - aligned).norm(dim=-1), mask
            )
            metrics["h_aligned_cos"] = self._masked_mean(self._cos(raw, aligned), mask)
        self._record(t, **metrics)
        self._trajectory("aligned", aligned, t, mask)
        self._trajectory("pooled", pooled, t)
        self._token_record(t, mask, pooling_residual_norm=residual.norm(dim=-1))
        self._current_h_curr = None

    @torch.no_grad()
    def _classifier_hook(self, module, inputs, output):
        if self._current_labels is None:
            return
        t = self._current_t
        z, logits = inputs[0].detach().float(), output.detach().float()
        labels = self._current_labels.to(logits.device)
        ce = functional.cross_entropy(logits, labels, reduction="none")
        probabilities = logits.softmax(-1)
        metrics = dict(
            ce=ce,
            accuracy=(logits.argmax(-1) == labels).float(),
            entropy=-(probabilities * logits.log_softmax(-1)).sum(-1),
        )
        # CE/softmax sees class differences: remove the common-logit direction
        # before finding the head row space. Bias cancels between loops.
        if self._head_version != module.weight._version:
            weight = module.weight.detach().float()
            centered = weight - weight.mean(0, keepdim=True)
            _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
            tolerance = (
                max(centered.shape) * torch.finfo(weight.dtype).eps * singular.max()
            )
            self._head_basis = vh[singular > tolerance]
            self._head_version = module.weight._version
        basis = self._head_basis
        assert basis is not None
        invisible = z - (z @ basis.T) @ basis
        metrics["head_invisible_state_fraction"] = self._ratio(
            invisible.square().sum(-1), z.square().sum(-1)
        )
        previous = self._previous.get("normalized")
        if previous is not None:
            dz = z - previous
            invisible_delta = dz - (dz @ basis.T) @ basis
            metrics["head_invisible_motion_fraction"] = self._ratio(
                invisible_delta.square().sum(-1), dz.square().sum(-1)
            )
            # Local first-order CE-insensitive motion, distinct from exact head nullspace.
            target = functional.one_hot(labels, logits.shape[-1])
            gradient = (probabilities - target) @ module.weight.detach().float()
            along_gradient = self._ratio(
                (dz * gradient).sum(-1).square(), gradient.square().sum(-1)
            )
            metrics["ce_tangent_motion_fraction"] = (
                1 - self._ratio(along_gradient, dz.square().sum(-1))
            ).clamp(0, 1)
            metrics["ce_change"] = ce - self._previous["ce"]
            metrics["ce_abs_change"] = metrics["ce_change"].abs()
            metrics["probability_delta"] = (
                probabilities - self._previous["probabilities"]
            ).norm(dim=-1)
        self._record(t, **metrics)
        self._trajectory("normalized", z, t)
        self._trajectory("logits", logits, t)
        self._previous["ce"] = ce
        self._previous["probabilities"] = probabilities

    @staticmethod
    def _stats(values: list[float | None]) -> dict:
        values = [value for value in values if value is not None]
        if not values:
            return {
                "mean": None,
                "std": None,
                "p10": None,
                "median": None,
                "p90": None,
                "count": 0,
            }
        x = torch.tensor(values, dtype=torch.float64)
        return dict(
            mean=x.mean().item(),
            std=x.std(correction=0).item(),
            p10=x.quantile(0.1).item(),
            median=x.quantile(0.5).item(),
            p90=x.quantile(0.9).item(),
            count=len(values),
        )

    def get_report(self) -> dict:
        """Return structured values without formatting or file writes."""
        loops: list[dict[str, Any]] = []
        for t in range(self._max_iterations):
            rows = [sample[t] for sample in self._rows.values() if t in sample]
            keys = sorted({key for row in rows for key in row})
            loops.append(
                dict(
                    t=t,
                    iteration=t + 1,
                    count=len(rows),
                    metrics={
                        key: self._stats(
                            [row[key] for row in rows if row.get(key) is not None]
                        )
                        for key in keys
                    },
                )
            )
        report = dict(
            schema_version=2,
            run_name=self._run_name,
            sample_count=self._sample_offset,
            sample_keys=dict(self._sample_keys),
            ordered_sample_fingerprint=hashlib.sha256(
                "".join(self._sample_keys.values()).encode()
            ).hexdigest(),
            loops=loops,
            samples={
                sid: {t: dict(row) for t, row in rows.items()}
                for sid, rows in self._rows.items()
            },
            token_samples={
                sid: {
                    t: {k: list(v) for k, v in row.items()} for t, row in rows.items()
                }
                for sid, rows in self._tokens.items()
            },
            classifier_visible_rank=None
            if self._head_basis is None
            else len(self._head_basis),
            measurement=self._measurement(),
            plateau=self._plateau(),
        )
        return report

    def _measurement(self) -> dict:
        return {
            "iteration_index_base": 1,
            "t_index_base": 0,
            "aggregation": "equal_example_weight_after_valid_token_mean",
            "dispersion": "between_examples_population_std_and_quantiles",
            "undefined_value": None,
            "first_raw_transition": "initial_input_to_loop_1",
            "token_sample_selection": {
                "method": "first_n",
                "limit": self._max_token_samples,
            },
            "sample_identity": "sha256(premise NUL hypothesis NUL normalized_label)",
            "execution": "observed_loops_only_no_halting_override",
            "rotation_scope": "loop_output_rotation_not_attention_rope_ablation",
            "energy_interpretation": "geometric_not_mutual_information",
            "definitions": {
                "norm": "L2(current)",
                "delta": "L2(current - previous)",
                "relative_delta": "L2(current - previous) / L2(previous)",
                "cos": "dot(current, previous) / (L2(current) * L2(previous))",
                "pooling_residual_fraction": "mean_token(||aligned - pooled||^2) / mean_token(||aligned||^2)",
                "pooling_cancelled_motion_fraction": "mean_token(||delta_aligned - mean_token(delta_aligned)||^2) / mean_token(||delta_aligned||^2)",
                "head_invisible_state_fraction": "||P_null(W_centered) normalized_pooled||^2 / ||normalized_pooled||^2",
                "head_invisible_motion_fraction": "||P_null(W_centered) delta_normalized_pooled||^2 / ||delta_normalized_pooled||^2",
                "ce_tangent_motion_fraction": "1 - dot(delta_z, grad_z_CE)^2 / (||delta_z||^2 * ||grad_z_CE||^2); local_first_order_only",
                "pre_rotation_delta": "mean_token(||inverse_R_step(raw_current) - raw_previous||)",
                "rotation_delta": "mean_token(||raw_current - inverse_R_step(raw_current)||)",
                "rotation_norm_error": "mean_token(abs(||raw_current|| - ||inverse_R_step(raw_current)||))",
                "h_aligned_delta": "mean_token(||raw_current - aligned_current||)",
                "ce": "cross_entropy_nats",
                "ce_abs_change": "abs(CE_current - CE_previous)",
                "probability_delta": "L2(softmax_current - softmax_previous)",
            },
        }

    def _plateau(self) -> dict:
        # Restrict every number to the SAME examples that reached all of loops 4-6.
        cohort = [
            rows
            for rows in self._rows.values()
            if all(t in rows and "ce" in rows[t] for t in (3, 4, 5))
        ]
        keys = (
            "ce_abs_change",
            "aligned_relative_delta",
            "pooled_relative_delta",
            "pooling_cancelled_motion_fraction",
            "head_invisible_motion_fraction",
            "probability_delta",
        )
        return dict(
            iterations=[4, 5, 6],
            count=len(cohort),
            metrics={
                key: self._stats(
                    [
                        rows[t][key]
                        for rows in cohort
                        for t in (4, 5)
                        if rows[t].get(key) is not None
                    ]
                )
                for key in keys
            },
        )

    def get_top_samples(self, metric: str = "final_ce", k: int = 10) -> list[int]:
        if metric not in ("final_ce", "ce_drop", "ce_rise", "ce_var"):
            raise ValueError(f"Unknown metric: {metric}")
        scores = {}
        for sid, rows in self._rows.items():
            ces = [
                row["ce"]
                for _, row in sorted(rows.items())
                if row.get("ce") is not None
            ]
            if not ces:
                continue
            scores[sid] = {
                "final_ce": ces[-1],
                "ce_drop": ces[0] - ces[-1],
                "ce_rise": ces[-1] - min(ces),
                "ce_var": sum((x - sum(ces) / len(ces)) ** 2 for x in ces) / len(ces),
            }[metric]
        return sorted(scores, key=lambda sample_id: scores[sample_id], reverse=True)[
            : max(0, k)
        ]
