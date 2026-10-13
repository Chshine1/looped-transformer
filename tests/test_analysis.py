import unittest
from dataclasses import replace

import torch

from looped_transformer.analysis.collect import (
    Observations,
    collect,
    cross_entropy,
    full_depth,
    gradients,
    summarize,
)
from looped_transformer.config import load_options
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.types.tokenizer import TokenizerInfo


class AnalysisTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(17)
        options = load_options("config.yaml").model
        options = replace(
            options,
            hidden_state_dimension=16,
            max_iterations=3,
            residual_scaling=replace(options.residual_scaling, expected_iterations=3),
            max_sequence_length=8,
            transformer=replace(
                options.transformer,
                attention_heads_count=2,
                swi_glu_ffn_dimension=24,
                gelu_ffn_dimension=24,
            ),
        )
        self.model = LoopedMNLI(options, TokenizerInfo(vocab_size=31, pad_token_id=0))
        self.ids = torch.randint(1, 31, (5, 8))
        self.mask = torch.ones_like(self.ids)
        self.mask[0, 4:] = 0
        self.labels = torch.tensor([0, 1, 2, 0, 1])

    def test_collection_preserves_modes_and_production_logits(self) -> None:
        self.model.train()
        self.model._halting_cell.eval()
        modes = [m.training for m in self.model.modules()]
        observed = collect(self.model, self.ids, self.mask, self.labels, 2)
        self.assertEqual(modes, [m.training for m in self.model.modules()])
        with full_depth(self.model), torch.no_grad():
            expected = self.model(self.ids, self.mask)["timed_logits"]
        torch.testing.assert_close(observed.logits, expected, atol=1e-6, rtol=1e-5)
        single_batch = collect(self.model, self.ids, self.mask, self.labels, 5)
        first = summarize(observed, self.model._classifier.weight)
        second = summarize(single_batch, self.model._classifier.weight)
        torch.testing.assert_close(torch.tensor(first.ce), torch.tensor(second.ce))
        self.assertEqual(sum(first.best_ce_depth_counts), 5)
        self.assertGreaterEqual(first.any_depth_correct_accuracy, max(first.accuracy))

    def test_microbatch_gradient_matches_direct_mean_loss(self) -> None:
        report = gradients(self.model, self.ids, self.mask, self.labels, 2)
        params = [
            p
            for name, p in self.model.named_parameters()
            if not name.startswith("_halting_cell")
        ]
        with full_depth(self.model):
            output = self.model(self.ids, self.mask)
            loss = cross_entropy(output["timed_logits"], self.labels).mean()
            grad = torch.autograd.grad(loss, params, allow_unused=True)
        expected = sum(g.square().sum() for g in grad if g is not None).sqrt().item()
        self.assertAlmostEqual(
            report.groups["all"].mean_objective_norm, expected, places=5
        )
        self.assertEqual(report.future_state_gradient_norm[-1], 0.0)
        self.assertTrue(all(v > 0 for v in report.local_state_gradient_norm))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))

    def test_modes_restored_on_exception(self) -> None:
        self.model.eval()
        with self.assertRaisesRegex(RuntimeError, "deliberate"), full_depth(self.model):
            raise RuntimeError("deliberate")
        self.assertTrue(all(not m.training for m in self.model.modules()))

    def test_classifier_nullspace_motion_is_not_logit_motion(self) -> None:
        weight = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        )
        features = torch.tensor([[[2.0, 0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0, 3.0]]])
        obs = Observations(
            features @ weight.T,
            features,
            torch.tensor([0]),
            torch.tensor([[1.5]]),
            torch.tensor([[1.5]]),
            {},
        )
        report = summarize(obs, weight)
        self.assertEqual(report.classifier_visible_step_energy_fraction, [0.0])
        self.assertEqual(report.prediction_flip_fraction, [0.0])
        self.assertEqual(report.ce[0], report.ce[1])
        self.assertGreater(report.normalized_relative_step[0], 0.0)


if __name__ == "__main__":
    unittest.main()
