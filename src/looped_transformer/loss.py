import torch
from torch import Tensor, nn
from torch.nn import functional

from looped_transformer.types.model import ModelTrainOutput


class LoopedMNLILoss(nn.Module):
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

    def forward(
        self,
        output: ModelTrainOutput,
        labels: Tensor,
    ) -> Tensor:
        # Keep reductions and logarithms in FP32 even when the forward pass uses AMP.
        logits = output["timed_logits"].float()
        probabilities = output["halt_probabilities"].float()
        time = logits.size(0)

        classes = logits.size(-1)
        cross_entropy = functional.cross_entropy(
            logits.reshape(-1, classes),
            labels.unsqueeze(0).expand(time, -1).reshape(-1),
            reduction="none",
        ).reshape(time, -1)

        loss_task = (probabilities * cross_entropy).sum(dim=0).mean()

        q = self._get_geometric_distribution(
            time, 1.0 / self._expected_time, logits.device, logits.dtype
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
