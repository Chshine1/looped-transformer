"""Collect trajectories using the production forward, without altering its math."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from looped_transformer.analysis.reports import (
    GradientGroupReport,
    GradientReport,
    ProbeReport,
    TrajectoryReport,
)
from looped_transformer.modules.model import LoopedMNLI


@contextmanager
def full_depth(model: LoopedMNLI) -> Iterator[None]:
    """All depths, dropout off; restore every module's mode, including on error."""
    modes = [(module, module.training) for module in model.modules()]
    model.eval()
    # Only the root controls early stopping. Children remain in evaluation mode.
    model.training = True
    try:
        yield
    finally:
        for module, mode in modes:
            module.training = mode


@dataclass
class Observations:
    logits: Tensor
    features: Tensor
    labels: Tensor
    token_steps: Tensor
    pooled_steps: Tensor
    component_updates: dict[str, Tensor]


def relative_step(current: Tensor, previous: Tensor) -> Tensor:
    dims = tuple(range(1, current.ndim))
    return (current - previous).square().sum(dims).sqrt() / previous.square().sum(
        dims
    ).sqrt().clamp_min(1e-12)


def collect(
    model: LoopedMNLI, ids: Tensor, mask: Tensor, labels: Tensor, batch_size: int
) -> Observations:
    logits_batches, feature_batches, token_batches, pooled_batches = [], [], [], []
    component_batches: dict[str, list[Tensor]] = {
        "block1": [],
        "block2": [],
        "injection": [],
    }
    device = next(model.parameters()).device
    with full_depth(model), torch.no_grad():
        for start in range(0, len(labels), batch_size):
            features: list[Tensor] = []
            pooled: list[Tensor] = []
            token_steps: list[Tensor] = []
            previous: Tensor | None = None
            component_updates: dict[str, list[Tensor]] = {
                key: [] for key in component_batches
            }

            def component_hook(
                module: nn.Module,
                args: Any,
                output: Tensor,
                component_updates: dict[str, list[Tensor]] = component_updates,
            ) -> None:
                if module is model._input_injection:
                    name = "injection"
                    valid = args[2].unsqueeze(-1)
                    iteration = len(component_updates["injection"]) + 1
                    updated = (
                        args[0]
                        + model.injection_scale(iteration)
                        * model._injection_gate
                        * output
                    )
                else:
                    name = "block1" if module is model._transformer_block1 else "block2"
                    valid = args[1].unsqueeze(-1)
                    updated = output
                component_updates[name].append(
                    relative_step(updated * valid, args[0] * valid)
                )

            def feature_hook(
                module: nn.Module, args: Any, features: list[Tensor] = features
            ) -> None:
                features.append(args[0].detach())

            def pooled_hook(
                module: nn.Module, args: Any, pooled: list[Tensor] = pooled
            ) -> None:
                pooled.append(args[0].detach())

            def token_hook(
                module: nn.Module, args: Any, token_steps: list[Tensor] = token_steps
            ) -> None:
                nonlocal previous
                current = args[1] * args[3].unsqueeze(-1)
                if previous is not None:
                    token_steps.append(relative_step(current, previous))
                previous = current

            handles = [
                model._classifier.register_forward_pre_hook(feature_hook),
                model._classify_normalization.register_forward_pre_hook(pooled_hook),
                model._halting_cell.register_forward_pre_hook(token_hook),
                model._transformer_block1.register_forward_hook(component_hook),
                model._transformer_block2.register_forward_hook(component_hook),
                model._input_injection.register_forward_hook(component_hook),
            ]
            try:
                out = model(
                    ids[start : start + batch_size].to(device),
                    mask[start : start + batch_size].to(device),
                )
                assert out["mode"] == "train"
                logits_batches.append(out["timed_logits"].cpu())
                feature_batches.append(torch.stack(features).cpu())
                for name, updates in component_updates.items():
                    component_batches[name].append(torch.stack(updates).cpu())
                token_batches.append(torch.stack(token_steps).cpu())
                pooled_batches.append(
                    torch.stack(
                        [relative_step(b, a) for a, b in pairwise(pooled)]
                    ).cpu()
                )
            finally:
                for handle in handles:
                    handle.remove()
    return Observations(
        torch.cat(logits_batches, 1),
        torch.cat(feature_batches, 1),
        labels,
        torch.cat(token_batches, 1),
        torch.cat(pooled_batches, 1),
        {name: torch.cat(batches, 1) for name, batches in component_batches.items()},
    )


def cross_entropy(logits: Tensor, labels: Tensor) -> Tensor:
    return F.cross_entropy(
        logits.flatten(0, 1), labels.repeat(logits.size(0)), reduction="none"
    ).view(logits.shape[:2])


def summarize(
    obs: Observations, weight: Tensor, compute_cost: float = 0.02
) -> TrajectoryReport:
    ce = cross_entropy(obs.logits, obs.labels)
    correct = obs.logits.argmax(-1) == obs.labels
    delta = ce[1:] - ce[:-1]
    steps = obs.features[1:] - obs.features[:-1]
    # Softmax ignores the common logit direction; use the centered weight row space.
    centered = weight.detach().float().cpu() - weight.detach().float().cpu().mean(0)
    _, singular, vh = torch.linalg.svd(centered, full_matrices=False)
    basis = vh[singular > singular.max() * 1e-6]
    visible = (steps @ basis.T).square().sum((1, 2)) / steps.square().sum(
        (1, 2)
    ).clamp_min(1e-12)
    times = torch.arange(1, len(ce) + 1)[:, None]
    return TrajectoryReport(
        examples=len(obs.labels),
        ce=ce.mean(1).tolist(),
        accuracy=correct.float().mean(1).tolist(),
        ce_change_standard_error=(
            delta.std(1, unbiased=False) / len(obs.labels) ** 0.5
        ).tolist(),
        improved_fraction=(delta < -0.01).float().mean(1).tolist(),
        worsened_fraction=(delta > 0.01).float().mean(1).tolist(),
        prediction_flip_fraction=(
            obs.logits[1:].argmax(-1) != obs.logits[:-1].argmax(-1)
        )
        .float()
        .mean(1)
        .tolist(),
        rescued_fraction=(correct[1:] & ~correct[:-1]).float().mean(1).tolist(),
        spoiled_fraction=(~correct[1:] & correct[:-1]).float().mean(1).tolist(),
        best_ce_depth_counts=torch.bincount(ce.argmin(0), minlength=len(ce)).tolist(),
        cost_optimal_depth_counts=torch.bincount(
            (ce + compute_cost * times).argmin(0), minlength=len(ce)
        ).tolist(),
        any_depth_correct_accuracy=correct.any(0).float().mean().item(),
        oracle_ce=ce.min(0).values.mean().item(),
        token_relative_step=obs.token_steps.mean(1).tolist(),
        pooled_relative_step=obs.pooled_steps.mean(1).tolist(),
        normalized_relative_step=torch.stack(
            [
                relative_step(b, a)
                for a, b in zip(obs.features[:-1], obs.features[1:], strict=True)
            ]
        )
        .mean(1)
        .tolist(),
        classifier_visible_step_energy_fraction=visible.tolist(),
        component_relative_updates={
            name: updates.mean(1).tolist()
            for name, updates in obs.component_updates.items()
        },
    )


def gradients(
    model: LoopedMNLI, ids: Tensor, mask: Tensor, labels: Tensor, batch_size: int
) -> GradientReport:
    """Exact gradient of the sample mean, accumulated in bounded microbatches."""
    named = [
        (n, p) for n, p in model.named_parameters() if not n.startswith("_halting_cell")
    ]
    names, params = zip(*named, strict=True)
    depth = model._model_options.max_iterations
    sums = [[torch.zeros_like(p, device="cpu") for p in params] for _ in range(depth)]
    state_local_energy = torch.zeros(depth, dtype=torch.float64)
    state_future_energy = torch.zeros(depth, dtype=torch.float64)
    state_dot = torch.zeros(depth, dtype=torch.float64)
    device = next(model.parameters()).device
    with full_depth(model), torch.enable_grad():
        for start in range(0, len(labels), batch_size):
            target = labels[start : start + batch_size].to(device)
            states: list[Tensor] = []

            def state_hook(
                module: nn.Module,
                args: Any,
                output: Tensor,
                states: list[Tensor] = states,
            ) -> None:
                states.append(output)

            handle = model._loop_normalization.register_forward_hook(state_hook)
            try:
                out = model(
                    ids[start : start + batch_size].to(device),
                    mask[start : start + batch_size].to(device),
                )
            finally:
                handle.remove()
            assert out["mode"] == "train"
            ce = cross_entropy(out["timed_logits"], target).mean(1)
            local = [torch.zeros_like(state) for state in states]
            future = [torch.zeros_like(state) for state in states]
            for t in range(depth):
                grads = torch.autograd.grad(
                    ce[t],
                    (*params, *states),
                    retain_graph=t < depth - 1,
                    allow_unused=True,
                )
                for accumulator, grad in zip(
                    sums[t], grads[: len(params)], strict=True
                ):
                    if grad is not None:
                        accumulator.add_(
                            grad.detach().cpu(), alpha=len(target) / len(labels)
                        )
                for s, grad in enumerate(grads[len(params) :]):
                    if grad is not None:
                        destination = local[s] if s == t else future[s]
                        destination.add_(
                            grad.detach(), alpha=len(target) / (len(labels) * depth)
                        )
            for s in range(depth):
                state_local_energy[s] += local[s].double().square().sum().cpu()
                state_future_energy[s] += future[s].double().square().sum().cpu()
                state_dot[s] += (local[s].double() * future[s].double()).sum().cpu()
    groups: dict[str, GradientGroupReport] = {}
    for group in ("all", "head", "backbone", "embedding", "injection", "gamma"):
        indices = [
            i
            for i, name in enumerate(names)
            if (
                group == "all"
                or (
                    group == "head"
                    and name.startswith(("_classifier", "_classify_normalization"))
                )
                or (
                    group == "backbone"
                    and name.startswith(("_transformer_block", "_loop_normalization"))
                )
                or (group == "embedding" and name.startswith("_token_embedding"))
                or (
                    group == "injection"
                    and name.startswith(("_injection_gate", "_input_injection"))
                )
                or (group == "gamma" and name == "_log_gamma_logit")
            )
        ]
        gram = torch.zeros(depth, depth, dtype=torch.float64)
        for i in indices:
            matrix = torch.stack([sums[t][i].flatten() for t in range(depth)]).double()
            gram += matrix @ matrix.T
        norms = gram.diag().clamp_min(0).sqrt()
        mean_norm = gram.mean().clamp_min(0).sqrt()
        groups[group] = GradientGroupReport(
            sum(params[i].numel() for i in indices),
            norms.tolist(),
            (gram / (norms[:, None] * norms[None, :]).clamp_min(1e-30)).tolist(),
            mean_norm.item(),
            (mean_norm / norms.mean().clamp_min(1e-30)).item(),
        )
    gamma_index = (
        names.index("_log_gamma_logit") if "_log_gamma_logit" in names else None
    )
    return GradientReport(
        len(labels),
        groups,
        [sums[t][gamma_index].item() for t in range(depth)]
        if gamma_index is not None
        else [],
        state_local_energy.sqrt().tolist(),
        state_future_energy.sqrt().tolist(),
        (
            state_dot
            / (state_local_energy * state_future_energy).sqrt().clamp_min(1e-30)
        ).tolist(),
    )


def probes(
    model: LoopedMNLI, train: Observations, evaluation: Observations, steps: int = 300
) -> ProbeReport:
    """Equal-budget shared vs per-depth head refits on frozen normalized features.

    Both initialize from the trained head. Evaluation labels never enter fitting.
    The fixed optimization budget is diagnostic, not a tuned capacity bound.
    """
    device = next(model.parameters()).device
    x, y = train.features.to(device), train.labels.to(device)
    ex, ey = evaluation.features.to(device), evaluation.labels.to(device)
    results = []
    with torch.enable_grad():
        for count in (1, x.size(0)):
            w = nn.Parameter(model._classifier.weight.detach().repeat(count, 1, 1))
            b = nn.Parameter(model._classifier.bias.detach().repeat(count, 1))
            optimizer = torch.optim.AdamW([w, b], lr=0.003, weight_decay=0.01)
            for _ in range(steps):
                logits = (
                    torch.einsum("tnd,tcd->tnc", x, w.expand(x.size(0), -1, -1))
                    + b[:, None]
                )
                loss = cross_entropy(logits, y).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                logits = (
                    torch.einsum("tnd,tcd->tnc", ex, w.expand(ex.size(0), -1, -1))
                    + b[:, None]
                )
                results.append(
                    (
                        cross_entropy(logits, ey).mean(1).tolist(),
                        (logits.argmax(-1) == ey).float().mean(1).tolist(),
                    )
                )
    return ProbeReport(
        len(y), steps, results[0][0], results[0][1], results[1][0], results[1][1]
    )
