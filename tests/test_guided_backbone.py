import unittest

import torch

from looped_transformer.domains.guided_backbone.loss import (
    ControllerWeightedLoopCrossEntropyLoss,
    ControllerWeightedLoopLossOptions,
)
from looped_transformer.domains.guided_backbone.model import (
    GuidanceWeightOptions,
    build_guidance_weights,
)


class ControllerWeightedLoopLossTests(unittest.TestCase):
    def test_guidance_weights_are_focused_but_blurred(self) -> None:
        weights = build_guidance_weights(
            torch.tensor([[2.0], [1.0], [0.0], [-1.0]]),
            GuidanceWeightOptions(
                temperature=0.35,
                neighbor_smoothing=0.2,
                uniform_smoothing=0.05,
            ),
        ).squeeze(1)

        torch.testing.assert_close(weights.sum(), torch.tensor(1.0))
        self.assertTrue(torch.all(weights > 0))
        self.assertGreater(weights[0].item(), weights[1].item())
        self.assertGreater(weights[1].item(), weights[2].item())
        self.assertLess(weights[0].item(), 0.8)

    def test_applies_detached_per_sample_loop_weights(self) -> None:
        logits = torch.tensor(
            [
                [[3.0, 0.0], [0.0, 3.0]],
                [[0.0, 3.0], [3.0, 0.0]],
            ],
            requires_grad=True,
        )
        weights = torch.tensor([[0.75, 0.25], [0.25, 0.75]], requires_grad=True)
        labels = torch.tensor([0, 1])
        loss = ControllerWeightedLoopCrossEntropyLoss(
            ControllerWeightedLoopLossOptions(
                type="controller_weighted_loop_cross_entropy",
                label_smoothing=0.0,
                time_rope_regularization_scale=0.0,
            )
        )

        actual = loss(
            {
                "mode": "train",
                "timed_logits": logits,
                "halt_probabilities": torch.zeros(2, 2),
                "loop_supervision_weights": weights,
            },
            labels,
        )
        actual.backward()

        self.assertIsNotNone(logits.grad)
        self.assertIsNone(weights.grad)

    def test_rejects_weights_that_do_not_sum_to_one(self) -> None:
        loss = ControllerWeightedLoopCrossEntropyLoss(
            ControllerWeightedLoopLossOptions(
                type="controller_weighted_loop_cross_entropy",
                time_rope_regularization_scale=0.0,
            )
        )
        with self.assertRaisesRegex(ValueError, "sum to one"):
            loss(
                {
                    "mode": "train",
                    "timed_logits": torch.zeros(2, 1, 3),
                    "halt_probabilities": torch.zeros(2, 1),
                    "loop_supervision_weights": torch.ones(2, 1),
                },
                torch.tensor([0]),
            )


if __name__ == "__main__":
    unittest.main()
