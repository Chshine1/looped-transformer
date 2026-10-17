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


class AnchoredSampledLoopCrossEntropyLoss(nn.Module):
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
