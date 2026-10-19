import unittest

import torch

from looped_transformer.evaluation.policies import LoopDiagnosticsPolicy
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.model import ModelForwardOutput


class LoopDiagnosticsPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.batch = MNLIBatch(
            premises=("p0", "p1", "p2", "p3"),
            hypotheses=("h0", "h1", "h2", "h3"),
            labels=("entailment", "contradiction", "neutral", "entailment"),
        )
        self.output: ModelForwardOutput = {
            "mode": "train",
            "timed_logits": torch.tensor(
                [
                    [
                        [5.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                        [0.0, 5.0, 0.0],
                    ],
                    [
                        [0.0, 5.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [0.0, 0.0, 5.0],
                        [0.0, 5.0, 0.0],
                    ],
                    [
                        [0.0, 5.0, 0.0],
                        [0.0, 5.0, 0.0],
                        [5.0, 0.0, 0.0],
                        [5.0, 0.0, 0.0],
                    ],
                ]
            ),
            "halt_probabilities": torch.full((3, 4), 1 / 3),
        }

    def test_reports_adaptive_headroom_over_fixed_depths(self) -> None:
        policy = LoopDiagnosticsPolicy(
            max_iterations=3,
            device=torch.device("cpu"),
            label_smoothing=0.1,
            shuffle_oracle_repeats=5,
            shuffle_seed=7,
        )

        policy.attend_batch(self.batch, self.output)
        diagnostics = policy.collect_diagnostics()

        self.assertEqual(diagnostics["per_sample_oracle"]["accuracy"], 1.0)
        self.assertEqual(diagnostics["final_loop"]["accuracy"], 0.5)
        self.assertGreater(
            diagnostics["adaptive_headroom"][
                "raw_cross_entropy_reduction_over_best_fixed"
            ],
            0.0,
        )
        self.assertEqual(
            diagnostics["adaptive_headroom"]["final_error_rescue_rate"], 0.5
        )
        self.assertEqual(
            diagnostics["per_loop"]["oracle_best_loop_histogram"], [1, 2, 1]
        )
        self.assertEqual(diagnostics["per_loop"]["loop"], [1, 2, 3])

    def test_shuffled_control_is_reproducible_after_reset(self) -> None:
        policy = LoopDiagnosticsPolicy(
            max_iterations=3,
            device=torch.device("cpu"),
            label_smoothing=0.0,
            shuffle_oracle_repeats=8,
            shuffle_seed=11,
        )

        policy.attend_batch(self.batch, self.output)
        first = policy.collect_diagnostics()["shuffled_label_oracle"]
        policy.reset()
        policy.attend_batch(self.batch, self.output)
        second = policy.collect_diagnostics()["shuffled_label_oracle"]

        self.assertEqual(first, second)
        self.assertEqual(len(first["accuracy_by_repeat"]), 8)

    def test_rejects_an_unexpected_loop_count(self) -> None:
        policy = LoopDiagnosticsPolicy(
            max_iterations=2,
            device=torch.device("cpu"),
            label_smoothing=0.0,
        )

        with self.assertRaisesRegex(ValueError, "Expected 2 loops"):
            policy.attend_batch(self.batch, self.output)


if __name__ == "__main__":
    unittest.main()
