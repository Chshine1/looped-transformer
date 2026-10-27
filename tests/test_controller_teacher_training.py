import unittest

import torch
from torch.nn import functional

from looped_transformer.domains.controller_teacher.controller_teacher_training import (
    ControllerTeacherLoss,
    ControllerTeacherLossInput,
    build_loop_quality_targets,
)


class ControllerTeacherLossTests(unittest.TestCase):
    def test_targets_prefer_low_loss_loops_but_retain_blurred_mass(self) -> None:
        timed_logits = torch.tensor(
            [
                [[0.0, 2.0]],
                [[1.0, 1.0]],
                [[3.0, 0.0]],
                [[0.0, 2.0]],
            ]
        )
        target = build_loop_quality_targets(
            timed_logits,
            torch.tensor([0]),
            temperature=0.1,
            neighbor_smoothing=0.2,
            uniform_smoothing=0.05,
        ).squeeze(1)

        torch.testing.assert_close(target.sum(), torch.tensor(1.0))
        self.assertTrue(torch.all(target > 0))
        self.assertAlmostEqual(target[2].item(), target[1].item())
        self.assertGreater(target[1].item(), target[0].item())
        self.assertGreater(target[2].item(), target[3].item())

    def test_matches_bce_with_smoothed_correctness_targets(self) -> None:
        timed_logits = torch.tensor(
            [
                [[2.0, 0.0], [3.0, 1.0]],
                [[0.0, 2.0], [1.0, 3.0]],
            ]
        )
        labels = torch.tensor([0, 1])
        selector_logits = torch.tensor([[0.2, -0.3], [1.1, 0.7]], requires_grad=True)
        expected_targets = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

        actual = ControllerTeacherLoss()(
            selector_logits,
            ControllerTeacherLossInput(timed_logits=timed_logits, labels=labels),
        )
        expected = functional.binary_cross_entropy_with_logits(
            selector_logits, expected_targets
        )

        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertIsNotNone(selector_logits.grad)

    def test_rejects_misaligned_labels(self) -> None:
        with self.assertRaisesRegex(ValueError, "labels must have shape"):
            ControllerTeacherLoss()(
                torch.zeros(2, 3),
                ControllerTeacherLossInput(
                    timed_logits=torch.zeros(2, 3, 4),
                    labels=torch.zeros(2, dtype=torch.long),
                ),
            )


if __name__ == "__main__":
    unittest.main()
