import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.types.model import ModelTrainOutput


def _cross_entropy_by_time(output: ModelTrainOutput, labels: Tensor) -> Tensor:
    """Return unreduced cross-entropy with shape (iterations, batch)."""
    logits = output["timed_logits"].float()
    iterations, _, classes = logits.shape
    return functional.cross_entropy(
        logits.reshape(-1, classes),
        labels.unsqueeze(0).expand(iterations, -1).reshape(-1),
        reduction="none",
    ).reshape(iterations, -1)


def _depth_weights(cross_entropy: Tensor, growth: float) -> Tensor:
    iterations = cross_entropy.size(0)
    indices = torch.arange(
        iterations, device=cross_entropy.device, dtype=cross_entropy.dtype
    )
    return torch.softmax(growth * indices, dim=0)


def _monotonic_loss(cross_entropy: Tensor) -> Tensor:
    if cross_entropy.size(0) < 2:
        return cross_entropy.new_zeros(())
    # Only improve a worse later step; never satisfy the constraint by making the
    # preceding step worse.
    return functional.relu(cross_entropy[1:] - cross_entropy[:-1].detach()).mean()


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


def _oracle_halting_distribution(
    cross_entropy: Tensor,
    compute_cost: float,
    temperature: float,
) -> Tensor:
    iterations = cross_entropy.size(0)
    time = torch.arange(
        1,
        iterations + 1,
        device=cross_entropy.device,
        dtype=cross_entropy.dtype,
    ).unsqueeze(1)
    cost = cross_entropy.detach() + compute_cost * time
    return torch.softmax(-cost / temperature, dim=0)


def _distribution_cross_entropy(target: Tensor, probabilities: Tensor) -> Tensor:
    return -(target * torch.log(probabilities.float().clamp_min(1e-8))).sum(0).mean()


class LastLoopCrossEntropyLoss(nn.Module):
    """Give every recurrent depth the same direct classification objective."""

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        return _cross_entropy_by_time(output, labels)[-1].mean()


class DepthWeightedCrossEntropyLoss(nn.Module):
    """Emphasize later depths while penalizing degradation between adjacent depths."""

    def __init__(self, depth_weight_growth: float, monotonic_loss_scale: float):
        super().__init__()
        self._depth_weight_growth = depth_weight_growth
        self._monotonic_loss_scale = monotonic_loss_scale

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        cross_entropy = _cross_entropy_by_time(output, labels)
        weights = _depth_weights(cross_entropy, self._depth_weight_growth)
        weighted_task = (weights[:, None] * cross_entropy).sum(0).mean()
        return weighted_task + self._monotonic_loss_scale * _monotonic_loss(
            cross_entropy
        )


class OracleHaltingLoss(nn.Module):
    """Fit the controller to the loss--compute optimum of a frozen backbone."""

    def __init__(self, compute_cost: float, target_temperature: float):
        super().__init__()
        self._compute_cost = compute_cost
        self._target_temperature = target_temperature

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        cross_entropy = _cross_entropy_by_time(output, labels)
        target = _oracle_halting_distribution(
            cross_entropy, self._compute_cost, self._target_temperature
        )
        return _distribution_cross_entropy(target, output["halt_probabilities"])


class AdaptiveJointLoss(nn.Module):
    """Jointly optimize expected task risk, compute, depth stability, and halting."""

    def __init__(
        self,
        compute_cost: float,
        target_temperature: float,
        controller_loss_scale: float,
        depth_loss_scale: float,
        depth_weight_growth: float,
        monotonic_loss_scale: float,
    ):
        super().__init__()
        self._compute_cost = compute_cost
        self._target_temperature = target_temperature
        self._controller_loss_scale = controller_loss_scale
        self._depth_loss_scale = depth_loss_scale
        self._depth_weight_growth = depth_weight_growth
        self._monotonic_loss_scale = monotonic_loss_scale

    def forward(self, output: ModelTrainOutput, labels: Tensor) -> Tensor:
        cross_entropy = _cross_entropy_by_time(output, labels)
        probabilities = output["halt_probabilities"].float()
        iterations = cross_entropy.size(0)
        time = torch.arange(
            1,
            iterations + 1,
            device=cross_entropy.device,
            dtype=cross_entropy.dtype,
        ).unsqueeze(1)

        expected_risk = (probabilities * cross_entropy).sum(0).mean()
        expected_compute = (probabilities * time).sum(0).mean()

        weights = _depth_weights(cross_entropy, self._depth_weight_growth)
        depth_task = (weights[:, None] * cross_entropy).sum(0).mean()
        depth_task = depth_task + self._monotonic_loss_scale * _monotonic_loss(
            cross_entropy
        )

        target = _oracle_halting_distribution(
            cross_entropy, self._compute_cost, self._target_temperature
        )
        controller_loss = _distribution_cross_entropy(target, probabilities)

        return (
            expected_risk
            + self._compute_cost * expected_compute
            + self._depth_loss_scale * depth_task
            + self._controller_loss_scale * controller_loss
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
