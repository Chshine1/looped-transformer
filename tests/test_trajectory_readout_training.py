import unittest
from typing import Any, cast

import torch

from looped_transformer.domains.clutrr.types import CLUTRRBatch
from looped_transformer.domains.trajectory_readout import (
    TrajectoryBatch,
    TrajectoryReadout,
    TrajectoryReadoutCollator,
    TrajectoryReadoutEvaluationPolicy,
    TrajectoryReadoutOptions,
)


class TrajectoryReadoutTrainingTests(unittest.TestCase):
    def test_collator_and_bidirectional_readout_train_together(self) -> None:
        source = CLUTRRBatch(
            ids=("one", "two"),
            stories=("story one", "story two"),
            queries=("query one", "query two"),
            labels=("aunt", "uncle"),
            depths=(2, 3),
        )
        batch = TrajectoryBatch(
            source=source,
            timed_hidden_states=torch.randn(4, 2, 3, 8),
            padding_mask=torch.tensor([[1, 1, 0], [1, 1, 1]]),
            final_logits=torch.randn(2, 18),
        )
        collated = TrajectoryReadoutCollator().collate_batch(
            batch, torch.device("cpu"), torch.float32
        )
        model = TrajectoryReadout(
            state_dimension=8,
            classes_count=18,
            options=TrajectoryReadoutOptions(
                architecture="bidirectional",
                encoding_dimension=12,
                dropout=0.0,
            ),
        )

        logits = model(*collated[0].inputs, None)

        self.assertEqual(tuple(logits.shape), (2, 18))
        self.assertEqual(collated[0].targets.labels.tolist(), [0, 17])
        logits.sum().backward()
        self.assertIsNotNone(model._readout.classifier_weight.grad)

    def test_evaluation_reports_readout_lift_and_depth_metrics(self) -> None:
        class Logger:
            def __init__(self) -> None:
                self.records = []

            def start(self) -> None:
                self.records.clear()

            def log_batch_evaluation(self, diagnostics: object) -> None:
                self.records.append(diagnostics)

        logger = Logger()
        policy = TrajectoryReadoutEvaluationPolicy(
            label_smoothing=0.0,
            logger=cast(Any, logger),
        )
        policy.reset(cast(Any, None))
        source = CLUTRRBatch(
            ids=("one", "two"),
            stories=("story one", "story two"),
            queries=("query one", "query two"),
            labels=("aunt", "uncle"),
            depths=(2, 3),
        )
        backbone_logits = torch.zeros(2, 18)
        backbone_logits[:, 1] = 2.0
        readout_logits = torch.zeros(2, 18)
        readout_logits[0, 0] = 3.0
        readout_logits[1, 17] = 3.0

        policy.attend_batch(
            TrajectoryBatch(
                source=source,
                timed_hidden_states=torch.empty(0),
                padding_mask=torch.empty(0),
                final_logits=backbone_logits,
            ),
            readout_logits,
        )
        summary = policy.collect_diagnostics()

        self.assertEqual(summary["readout"]["accuracy"], 1.0)
        self.assertEqual(summary["backbone_final"]["accuracy"], 0.0)
        self.assertEqual(summary["lift_over_backbone"]["accuracy_gain"], 1.0)
        self.assertEqual(summary["by_reasoning_depth"]["2"]["sample_count"], 1)
        self.assertEqual(summary["by_reasoning_depth"]["3"]["sample_count"], 1)
        self.assertEqual(len(logger.records), 1)

    def test_causal_readout_uses_final_prefix_for_training(self) -> None:
        model = TrajectoryReadout(
            state_dimension=8,
            classes_count=18,
            options=TrajectoryReadoutOptions(
                architecture="causal",
                encoding_dimension=12,
                dropout=0.0,
            ),
        )

        logits = model(
            torch.randn(4, 2, 3, 8),
            torch.ones(2, 3, dtype=torch.long),
            None,
        )

        self.assertEqual(tuple(logits.shape), (2, 18))


if __name__ == "__main__":
    unittest.main()
