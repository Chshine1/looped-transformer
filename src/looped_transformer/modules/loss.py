import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.types.model import ModelTrainOutput


def _cross_entropy_by_time(
    output: ModelTrainOutput, labels: Tensor, label_smoothing: float = 0.0
) -> Tensor:
    """Return unreduced cross-entropy with shape (iterations, batch)."""
    logits = output["timed_logits"].float()
    iterations, _, classes = logits.shape
    return functional.cross_entropy(
        logits.reshape(-1, classes),
        labels.unsqueeze(0).expand(iterations, -1).reshape(-1),
        reduction="none",
        label_smoothing=label_smoothing,
    ).reshape(iterations, -1)


def get_loop_metrics(output: ModelTrainOutput, labels: Tensor) -> tuple[Tensor, Tensor]:
    """Return mean cross-entropy per loop and mean expected iteration count."""
    cross_entropy = _cross_entropy_by_time(output, labels)
    probabilities = output["halt_probabilities"].float()
    time = torch.arange(
        1,
        cross_entropy.size(0) + 1,
        device=cross_entropy.device,
        dtype=cross_entropy.dtype,
    ).unsqueeze(1)
    return cross_entropy.mean(dim=1), (probabilities * time).sum(0).mean()


class SampledLoopCrossEntropyLoss(nn.Module):
    """Supervise one independently sampled recurrent depth per example."""

    def __init__(
        self,
        loops: tuple[int, ...],
        weights: tuple[float, ...],
        time_rope_regularization_scale: float = 0.0,
    ):
        super().__init__()
        if not loops or len(loops) != len(weights):
            raise ValueError("loops and weights must have the same nonzero length")
        if any(loop <= 0 for loop in loops) or len(set(loops)) != len(loops):
            raise ValueError("loops must contain unique positive 1-based indices")
        weight_tensor = torch.tensor(weights, dtype=torch.float32)
        if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
            raise ValueError("weights must be finite and nonnegative")
        if weight_tensor.sum() <= 0:
            raise ValueError("at least one sampled-loop weight must be positive")
        if not 0 <= time_rope_regularization_scale < float("inf"):
            raise ValueError(
                "time_rope_regularization_scale must be finite and nonnegative"
            )

        self._loops = tuple(loop - 1 for loop in loops)
        self.register_buffer("_weights", weight_tensor / weight_tensor.sum())
        self._time_rope_regularization_scale = time_rope_regularization_scale

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        cross_entropy = _cross_entropy_by_time(output, labels)
        if max(self._loops) >= cross_entropy.size(0):
            raise ValueError(
                f"sampled loop {max(self._loops) + 1} exceeds model loop count "
                f"{cross_entropy.size(0)}"
            )

        sampled_candidates = torch.multinomial(
            self._weights, labels.numel(), replacement=True
        )
        loop_indices = torch.tensor(
            self._loops, device=cross_entropy.device, dtype=torch.long
        )[sampled_candidates]
        batch_indices = torch.arange(labels.numel(), device=cross_entropy.device)
        task_loss = cross_entropy[loop_indices, batch_indices].mean()

        if self._time_rope_regularization_scale == 0:
            return task_loss
        return task_loss + (
            self._time_rope_regularization_scale * output["time_rope_regularization"]
        )


class AnchoredSampledLoopCrossEntropyLoss(nn.Module):
    """Anchor the final loop and weakly supervise one earlier loop per example."""

    def __init__(
        self,
        sampled_loops: tuple[int, ...],
        sampled_weights: tuple[float, ...],
        sampled_loss_scale: float,
        label_smoothing: float = 0.0,
        time_rope_regularization_scale: float = 0.0,
    ):
        super().__init__()
        if not sampled_loops or len(sampled_loops) != len(sampled_weights):
            raise ValueError(
                "sampled_loops and sampled_weights must have the same nonzero length"
            )
        if any(loop <= 0 for loop in sampled_loops) or len(set(sampled_loops)) != len(
            sampled_loops
        ):
            raise ValueError(
                "sampled_loops must contain unique positive 1-based indices"
            )
        weight_tensor = torch.tensor(sampled_weights, dtype=torch.float32)
        if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
            raise ValueError("sampled_weights must be finite and nonnegative")
        if weight_tensor.sum() <= 0:
            raise ValueError("at least one sampled-loop weight must be positive")
        if not 0 <= sampled_loss_scale < float("inf"):
            raise ValueError("sampled_loss_scale must be finite and nonnegative")
        if not 0 <= label_smoothing < 1:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0 <= time_rope_regularization_scale < float("inf"):
            raise ValueError(
                "time_rope_regularization_scale must be finite and nonnegative"
            )

        self._sampled_loops = tuple(loop - 1 for loop in sampled_loops)
        self.register_buffer("_sampled_weights", weight_tensor / weight_tensor.sum())
        self._sampled_loss_scale = sampled_loss_scale
        self._label_smoothing = label_smoothing
        self._time_rope_regularization_scale = time_rope_regularization_scale

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        cross_entropy = _cross_entropy_by_time(
            output, labels, label_smoothing=self._label_smoothing
        )
        if max(self._sampled_loops) >= cross_entropy.size(0) - 1:
            raise ValueError(
                "sampled loops must be earlier than the final model loop "
                f"{cross_entropy.size(0)}"
            )

        sampled_candidates = torch.multinomial(
            self._sampled_weights, labels.numel(), replacement=True
        )
        loop_indices = torch.tensor(
            self._sampled_loops, device=cross_entropy.device, dtype=torch.long
        )[sampled_candidates]
        batch_indices = torch.arange(labels.numel(), device=cross_entropy.device)
        sampled_loss = cross_entropy[loop_indices, batch_indices].mean()
        task_loss = cross_entropy[-1].mean() + self._sampled_loss_scale * sampled_loss

        if self._time_rope_regularization_scale == 0:
            return task_loss
        return task_loss + (
            self._time_rope_regularization_scale * output["time_rope_regularization"]
        )


class LoopedMNLILoss(nn.Module):
    """Legacy PonderNet-style objective retained for existing configurations."""

    def __init__(
        self,
        expected_time: int,
        halt_scale: float,
        aux_scale: float,
    ):
        super().__init__()
        self._expected_time = expected_time
        self._halt_scale = halt_scale
        self._aux_scale = aux_scale

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        cross_entropy = _cross_entropy_by_time(output, labels)
        probabilities = output["halt_probabilities"].float()
        time = cross_entropy.size(0)

        loss_task = (probabilities * cross_entropy).sum(dim=0).mean()
        q = self._get_geometric_distribution(
            time,
            1.0 / self._expected_time,
            cross_entropy.device,
            cross_entropy.dtype,
        )
        loss_halt = (
            (
                probabilities
                * (torch.log(probabilities + 1e-8) - torch.log(q[:, None] + 1e-8))
            )
            .sum(0)
            .mean()
        )
        loss_aux = cross_entropy.mean()
        return loss_task + self._halt_scale * loss_halt + self._aux_scale * loss_aux

    @staticmethod
    def _get_geometric_distribution(
        time: int, lam: float, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        q = lam * (1 - lam) ** torch.arange(time, device=device, dtype=dtype)
        q[-1] = (1 - lam) ** (time - 1)
        return q / q.sum()
