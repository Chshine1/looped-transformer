import unittest

import torch

from looped_transformer.modules.loss import (
    AdaptiveJointLoss,
    DepthWeightedCrossEntropyLoss,
    MeanLoopCrossEntropyLoss,
    OracleHaltingLoss,
)
from looped_transformer.types.model import ModelTrainOutput


def _make_output() -> tuple[ModelTrainOutput, torch.Tensor, torch.Tensor]:
    logits = torch.randn(4, 3, 3, requires_grad=True)
    hazard_logits = torch.randn(4, 3, requires_grad=True)
    hazards = torch.sigmoid(hazard_logits)
    survivals = torch.cat(
        [torch.ones(1, 3), torch.cumprod(1 - hazards[:-1], dim=0)], dim=0
    )
    final_hazards = torch.cat([hazards[:-1], torch.ones(1, 3)], dim=0)
    probabilities = final_hazards * survivals
    output: ModelTrainOutput = {
        "mode": "train",
        "timed_logits": logits,
        "halt_probabilities": probabilities,
    }
    return output, logits, hazard_logits


class StageLossTest(unittest.TestCase):
    def setUp(self) -> None:
        self.labels = torch.tensor([0, 1, 2])

    def test_backbone_losses_do_not_use_halting_probabilities(self) -> None:
        for loss in (
            MeanLoopCrossEntropyLoss(),
            DepthWeightedCrossEntropyLoss(0.5, 0.2),
        ):
            with self.subTest(loss=type(loss).__name__):
                output, logits, hazards = _make_output()
                value = loss(output, self.labels)
                value.backward()
                self.assertIsNotNone(logits.grad)
                self.assertIsNone(hazards.grad)

    def test_controller_loss_uses_only_halting_probabilities(self) -> None:
        output, logits, hazards = _make_output()

        value = OracleHaltingLoss(0.02, 0.1)(output, self.labels)
        value.backward()

        self.assertIsNone(logits.grad)
        self.assertIsNotNone(hazards.grad)

    def test_joint_loss_updates_logits_and_halting_probabilities(self) -> None:
        output, logits, hazards = _make_output()
        loss = AdaptiveJointLoss(0.02, 0.1, 0.1, 0.1, 0.5, 0.2)

        value = loss(output, self.labels)
        value.backward()

        self.assertTrue(torch.isfinite(value))
        self.assertIsNotNone(logits.grad)
        self.assertIsNotNone(hazards.grad)


if __name__ == "__main__":
    unittest.main()
