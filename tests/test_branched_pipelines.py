import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import torch
from torch import Tensor, nn

from looped_transformer.core.abstractions.training import CollatedBatch
from looped_transformer.core.abstractions.typed_module import (
    TypedBranchedModule,
    TypedModule,
)
from looped_transformer.core.evaluation.evaluator import (
    EvaluationCheckpointOptions,
    Evaluator,
)
from looped_transformer.core.modules.branched_module import BranchedModule
from looped_transformer.core.training.model_trainer import ModelTrainer


class _Scale(nn.Module):
    def __init__(self, factor: float) -> None:
        super().__init__()
        self.factor = nn.Parameter(torch.tensor(factor))

    def forward(self, value: Tensor) -> Tensor:
        return value * self.factor


class _TaskModel(BranchedModule[Tensor, Tensor]):
    def __init__(self) -> None:
        super().__init__()
        self.scale = self._create_branched(
            TypedModule(_Scale(2.0)),
            {"side": TypedModule(_Scale(3.0))},
        )

    def _forward_task(self, value: Tensor) -> Tensor:
        return self.scale(value)


class _SquaredLoss(nn.Module):
    def forward(self, output: Tensor, target: Tensor) -> Tensor:
        return (output - target).square()


class _CheckpointLoader:
    def load(self, _path: object) -> None:
        pass

    def load_model(self, _model: TypedBranchedModule) -> None:
        pass

    def load_training_state(self) -> object:
        raise AssertionError("no checkpoint should be resumed")

    def load_components(self, _components: object) -> None:
        raise AssertionError("no checkpoint should be resumed")

    def dispose(self) -> None:
        pass


class _Policy:
    def __init__(self) -> None:
        self.attended: list[tuple[str, float]] = []

    def reset(self, _model: object) -> None:
        self.attended.clear()

    def attend_batch(self, batch: str, output: Tensor) -> None:
        self.attended.append((batch, output.item()))

    def close(self) -> None:
        pass


class _Collator:
    def collate_batch(
        self, _batch: str, _device: torch.device, _dtype: torch.dtype
    ) -> list[CollatedBatch[str, Tensor, Tensor]]:
        return [
            CollatedBatch(
                batch="main sample",
                task=None,
                inputs=(torch.tensor(1.0),),
                targets=torch.tensor(0.0),
                samples_count=1,
                loss_scale=2.0,
            ),
            CollatedBatch(
                batch="side samples",
                task="side",
                inputs=(torch.tensor(1.0),),
                targets=torch.tensor(0.0),
                samples_count=2,
                loss_scale=0.5,
            ),
        ]


class _OneBatchProvider:
    def __init__(self) -> None:
        self._pending = False

    def reset_source(self, shuffle: bool, seed: int | None) -> None:
        del shuffle, seed
        self._pending = True

    def fetch(self) -> str | None:
        if not self._pending:
            return None
        self._pending = False
        return "mixed batch"

    def get_samples_count(self) -> int:
        return 3

    def get_batch_size(self) -> int:
        return 3

    def close(self) -> None:
        pass


def _runtime_options() -> tuple[Any, Any]:
    device = cast(
        Any,
        SimpleNamespace(device=torch.device("cpu")),
    )
    precision = cast(
        Any,
        SimpleNamespace(
            dtype=torch.float32,
            amp_enabled=False,
            amp_dtype=torch.float16,
            autocast=lambda _device: nullcontext(),
        ),
    )
    return device, precision


class BranchedPipelineTests(unittest.TestCase):
    def test_evaluator_dispatches_tasks_and_aligned_source_batches(self) -> None:
        model = TypedBranchedModule(_TaskModel())
        policy = _Policy()
        device, precision = _runtime_options()
        evaluator = Evaluator(
            model,
            cast(Any, _Collator()),
            cast(Any, policy),
            cast(Any, _OneBatchProvider()),
            cast(Any, _CheckpointLoader()),
            device,
            precision,
            EvaluationCheckpointOptions(path=Path(".")),
        )

        evaluator.open()
        try:
            outputs = evaluator.evaluate_batch("mixed batch")
        finally:
            evaluator.close()

        self.assertEqual([output.item() for output in outputs], [2.0, 3.0])
        self.assertEqual(
            policy.attended,
            [("main sample", 2.0), ("side samples", 3.0)],
        )

    def test_trainer_weights_task_losses_by_micro_batch_sample_count(self) -> None:
        model = TypedBranchedModule(_TaskModel())
        provider = _OneBatchProvider()
        stage_options = SimpleNamespace(
            epochs=1,
            max_gradient_norm=100.0,
            frozen_patterns=r"$^",
            evaluation_patterns=r"$^",
            save_every_epochs=100,
            gradient_accumulation_steps=1,
        )
        stage = cast(
            Any,
            SimpleNamespace(
                id="test",
                stage_options=stage_options,
                data_provider=provider,
                loss=TypedModule(_SquaredLoss()),
                optimizer_factory=lambda named: torch.optim.SGD(
                    [parameter for _, parameter in named], lr=0.1
                ),
                scheduler_factory=lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
                    optimizer, lambda _step: 1.0
                ),
            ),
        )
        device, precision = _runtime_options()
        reproducibility = cast(
            Any,
            SimpleNamespace(seed=0, activate=lambda _device: None),
        )
        trainer = ModelTrainer(
            model,
            cast(Any, _Collator()),
            stage,
            None,
            cast(Any, SimpleNamespace(save_epoch=lambda *_args: None)),
            cast(Any, _CheckpointLoader()),
            device,
            precision,
            reproducibility,
            cast(Any, SimpleNamespace(log_every_steps=100)),
        )

        trainer.train()

        parameters = dict(model.torch.named_parameters())
        self.assertAlmostEqual(parameters["scale.main.factor"].item(), 2.0 - 0.8 / 3)
        self.assertAlmostEqual(
            parameters["scale.branches.side.factor"].item(), 2.8, places=6
        )


if __name__ == "__main__":
    unittest.main()
