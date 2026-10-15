import re
import unittest
from dataclasses import replace
from pathlib import Path

import torch

from looped_transformer.config import load_options
from looped_transformer.config.model import TransformerOptions
from looped_transformer.modules.loss import LastLoopCrossEntropyLoss
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.modules.residual_scale import ResidualScale
from looped_transformer.stages.looped_mnli import (
    LastLoopLossOptions,
    _build_loss,
    _get_parameter_groups,
)
from looped_transformer.types.tokenizer import TokenizerInfo


class ArchitectureTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        options = load_options(Path(__file__).resolve().parents[1] / "config.yaml")
        self.options = replace(
            options.model,
            hidden_state_dimension=8,
            transformer=TransformerOptions(16, 16, 2),
            max_iterations=3,
            max_sequence_length=4,
            dropout=0.0,
        )
        self.model = LoopedMNLI(self.options, TokenizerInfo(20, 0))

    def test_residual_scale_initialization_bounds_and_gradient(self):
        scale = ResidualScale()
        self.assertEqual(scale().item(), 1.0)
        scale().backward()
        self.assertGreater(scale.logit.grad.item(), 0)
        for value in (-10.0, 10.0):
            with torch.no_grad():
                scale.logit.fill_(value)
            self.assertGreater(scale().item(), 0)
            self.assertLess(scale().item(), 2)

    def test_regularizer_matches_actual_time_rotation(self):
        attn = self.model._attn_res
        with torch.no_grad():
            for projection in (attn._q_proj, attn._k_proj):
                blocks = []
                for _ in range(2):
                    a, b = torch.randn(2, 2), torch.randn(2, 2)
                    blocks.append(
                        torch.cat([torch.cat([a, -b], 1), torch.cat([b, a], 1)], 0)
                    )
                projection.weight.copy_(torch.block_diag(*blocks))
                projection.bias.zero_()
        self.assertLess(attn.time_rope_regularization().item(), 1e-12)
        x = torch.randn(2, 4, 8)
        for t in (-3, -1, 1, 2, 3):
            rotate = self.model._alignment_rotation
            for projection in (attn._q_proj, attn._k_proj):
                torch.testing.assert_close(
                    projection(rotate(t, x)), rotate(t, projection(x))
                )
        # Cross-head mixing, non-complex-linear mixing, and bias each violate it.
        for row, col in ((0, 4), (0, 0)):
            with torch.no_grad():
                attn._q_proj.weight[row, col] += 1
            penalty = attn.time_rope_regularization()
            self.assertGreater(penalty.item(), 0)
            penalty.backward()
            self.assertGreater(attn._q_proj.weight.grad[row, col].item(), 0)
            attn.zero_grad()
            with torch.no_grad():
                attn._q_proj.weight[row, col] -= 1
        with torch.no_grad():
            attn._k_proj.bias[0] = 1
        self.assertGreater(attn.time_rope_regularization().item(), 0)

    def test_history_and_all_residual_branches_receive_task_gradients(self):
        lengths = []
        handle = self.model._attn_res.register_forward_pre_hook(
            lambda module, args: lengths.append(len(args[1]))
        )
        output = self.model(
            torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]]),
            torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]]),
        )
        handle.remove()
        self.assertEqual(lengths, [0, 1, 2])
        labels = torch.tensor([0, 2])
        task = LastLoopCrossEntropyLoss(False)(output, labels)
        regularized = _build_loss(
            self.options.training.stage_templates["backbone_warmup"].loss
        )(output, labels)
        torch.testing.assert_close(
            regularized, task + 0.01 * output["time_rope_regularization"]
        )
        task.backward()
        scales = [
            (n, p) for n, p in self.model.named_parameters() if "residual_scale" in n
        ]
        self.assertEqual(len(scales), 5)
        for name, parameter in [
            *scales,
            ("q", self.model._attn_res._q_proj.weight),
            ("k", self.model._attn_res._k_proj.weight),
        ]:
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)
        self.model.eval()
        with torch.no_grad():
            result = self.model(torch.ones(2, 4, dtype=torch.long), torch.ones(2, 4))
        self.assertEqual(result["logits"].shape, (2, 3))

    def test_all_stage_patterns_cover_trainable_parameters(self):
        for stage in self.options.training.stage_templates.values():
            parameters = [
                (n, p)
                for n, p in self.model.named_parameters()
                if not re.match(stage.common.frozen_patterns, n)
            ]
            groups = _get_parameter_groups(parameters, stage.optimizer)
            grouped = [id(p) for g in groups for p in g["params"]]
            self.assertEqual(len(grouped), len(set(grouped)))
            self.assertEqual(set(grouped), {id(p) for _, p in parameters})
            for pattern in stage.optimizer.learning_rates:
                self.assertTrue(
                    any(
                        re.fullmatch(pattern, n.rpartition(".")[0])
                        for n, _ in parameters
                    ),
                    pattern,
                )
            for group in groups:
                for parameter in group["params"]:
                    if parameter.ndim < 2:
                        self.assertEqual(group["weight_decay"], 0)
            optimizer = torch.optim.AdamW(groups)
            optimizer.zero_grad()

    def test_regularization_options_validation_and_legacy_default(self):
        for coefficient in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                LastLoopLossOptions(False, coefficient)
        self.assertEqual(LastLoopLossOptions(False).time_rope_regularization_scale, 0)


if __name__ == "__main__":
    unittest.main()
