import math
import re
import unittest
from dataclasses import replace
from typing import cast

import torch

from looped_transformer.config import load_options
from looped_transformer.config.model import ResidualScalingOptions
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.modules.multi_head_attention import RoPEMultiHeadAttention
from looped_transformer.stages import build_stage_templates
from looped_transformer.types.tokenizer import TokenizerInfo


class _StubDataReader:
    def count_examples(self, split: object) -> int:
        del split
        return 320


class RoPEMultiHeadAttentionTest(unittest.TestCase):
    def test_self_attention_preserves_shape(self) -> None:
        attention = RoPEMultiHeadAttention(32, 4, 16, 0.0)
        inputs = torch.randn(2, 7, 32)
        mask = torch.ones(2, 7, dtype=torch.long)

        output = attention(inputs, inputs, mask)

        self.assertEqual(output.shape, inputs.shape)
        self.assertTrue(torch.isfinite(output).all())

    def test_cross_attention_accepts_distinct_sequence_lengths(self) -> None:
        attention = RoPEMultiHeadAttention(32, 4, 16, 0.0)
        query = torch.randn(2, 5, 32)
        memory = torch.randn(2, 7, 32)
        memory_mask = torch.ones(2, 7, dtype=torch.long)

        output = attention(query, memory, memory_mask)

        self.assertEqual(output.shape, query.shape)

    def test_rope_tables_follow_module_dtype(self) -> None:
        attention = RoPEMultiHeadAttention(32, 4, 16, 0.0).to(dtype=torch.float64)

        self.assertEqual(attention._rope_cos.dtype, torch.float64)
        self.assertEqual(attention._rope_sin.dtype, torch.float64)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_rope_tables_follow_module_device(self) -> None:
        attention = RoPEMultiHeadAttention(32, 4, 16, 0.0).cuda()
        inputs = torch.randn(2, 7, 32, device="cuda")
        mask = torch.ones(2, 7, dtype=torch.long, device="cuda")

        output = attention(inputs, inputs, mask)

        self.assertEqual(output.device.type, "cuda")
        self.assertEqual(attention._rope_cos.device.type, "cuda")


class LoopedMNLITest(unittest.TestCase):
    def setUp(self) -> None:
        options = load_options("config.yaml").model
        self.model = LoopedMNLI(options, TokenizerInfo(vocab_size=101, pad_token_id=0))
        self.input_ids = torch.randint(1, 101, (2, 8))
        self.padding_mask = torch.ones(2, 8, dtype=torch.long)

    def test_training_forward_shapes_and_halting_distribution(self) -> None:
        self.model.train()

        output = self.model(self.input_ids, self.padding_mask)

        self.assertEqual(output["timed_logits"].shape, (6, 2, 3))
        self.assertEqual(output["halt_probabilities"].shape, (6, 2))
        torch.testing.assert_close(
            output["halt_probabilities"].sum(dim=0), torch.ones(2)
        )

    def test_evaluation_forward_shape(self) -> None:
        self.model.eval()

        with torch.inference_mode():
            output = self.model(self.input_ids, self.padding_mask)

        self.assertEqual(output["logits"].shape, (2, 3))

    def test_static_scale_is_shared_by_all_branches_and_depths(self) -> None:
        scales = []
        injection_inputs = []
        injection_outputs = []
        block_inputs = []

        def block_hook(module, args):
            scales.append(args[2].item())
            if module is self.model._transformer_block1:
                block_inputs.append(args[0].detach().clone())

        def injection_hook(module, args, output):
            injection_inputs.append(args[0].detach().clone())
            injection_outputs.append(output.detach().clone())

        handles = [
            self.model._transformer_block1.register_forward_pre_hook(block_hook),
            self.model._transformer_block2.register_forward_pre_hook(block_hook),
            self.model._input_injection.register_forward_hook(injection_hook),
        ]
        try:
            self.model.train()
            with torch.no_grad():
                self.model._injection_gate.fill_(0.3)
                self.model(self.input_ids, self.padding_mask)
            expected = 1 / (6 * math.sqrt(2))
            self.assertEqual(len(scales), 12)
            for scale in scales:
                self.assertAlmostEqual(scale, expected)
            for t in range(5):
                torch.testing.assert_close(
                    block_inputs[t + 1],
                    injection_inputs[t] + expected * 0.3 * injection_outputs[t],
                )
            self.assertNotIn("_log_gamma_logit", dict(self.model.named_parameters()))
        finally:
            for handle in handles:
                handle.remove()

    def test_legacy_scaling_remains_available_for_old_checkpoints(self) -> None:
        options = replace(
            load_options("config.yaml").model, residual_scaling=ResidualScalingOptions()
        )
        model = LoopedMNLI(options, TokenizerInfo(vocab_size=101, pad_token_id=0))
        for t in range(6):
            self.assertAlmostEqual(model.residual_scale(t).item(), 0.75**t / 2)
            self.assertEqual(model.injection_scale(t).item(), 1.0)

    def test_invalid_expected_iterations_are_rejected(self) -> None:
        for value in (0, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                ResidualScalingOptions(mode="static", expected_iterations=value)
        options = replace(
            load_options("config.yaml").model,
            residual_scaling=ResidualScalingOptions(
                mode="static", expected_iterations=7
            ),
        )
        with self.assertRaises(ValueError):
            LoopedMNLI(options, TokenizerInfo(vocab_size=101, pad_token_id=0))


class TrainingConfigurationTest(unittest.TestCase):
    def test_each_stage_covers_all_trainable_parameters(self) -> None:
        options = load_options("config.yaml")
        self.assertEqual(options.model.training.stages, ["backbone_warmup"])
        model = LoopedMNLI(options.model, TokenizerInfo(vocab_size=101, pad_token_id=0))
        stages = build_stage_templates(
            replace(
                options.model.training,
                stages=list(options.model.training.stage_templates),
            ),
            cast(MNLIDataReader, _StubDataReader()),
            options.data,
        )

        self.assertEqual(
            [stage.id for stage in stages],
            ["backbone_warmup", "depth_training", "controller_training", "joint"],
        )
        for stage in stages:
            with self.subTest(stage=stage.id):
                for name, parameter in model.named_parameters():
                    parameter.requires_grad = not bool(
                        re.match(stage.frozen_patterns, name)
                    )
                trainable_parameters = [
                    item for item in model.named_parameters() if item[1].requires_grad
                ]
                optimizer = stage.optimizer_factory(iter(trainable_parameters))
                optimized_count = sum(
                    len(group["params"]) for group in optimizer.param_groups
                )
                self.assertEqual(optimized_count, len(trainable_parameters))
                for group in optimizer.param_groups:
                    for parameter in group["params"]:
                        expected_decay = 0.01 if parameter.ndim >= 2 else 0.0
                        self.assertEqual(group["weight_decay"], expected_decay)
                scheduler = stage.scheduler_factory(optimizer)
                for _ in range(3):
                    optimizer.step()
                    scheduler.step()
                self.assertTrue(all(rate > 0 for rate in scheduler.get_last_lr()))


if __name__ == "__main__":
    unittest.main()
