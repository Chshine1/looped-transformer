"""Command-line interface for the MNLI application."""

import argparse
import json
from pathlib import Path

from torch import Tensor

from looped_transformer.app.config import load_options
from looped_transformer.app.container import build_container, configure_evaluation
from looped_transformer.core.abstractions.model import ModelForwardOutput, TypedModule
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.core.nn.options import LoopOptions
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.mnli.batch import MNLIBatch
from looped_transformer.domains.mnli.collator import MNLICollator
from looped_transformer.domains.mnli.data import MNLIDataOptions, MNLIDataProvider
from looped_transformer.domains.mnli.evaluation import EvaluationOptions
from looped_transformer.domains.mnli.model import LoopedMNLI, ModelOptions
from looped_transformer.domains.mnli.policies.loop_diagnostics import (
    LoopDiagnosticsOptions,
    LoopDiagnosticsPolicy,
)
from looped_transformer.domains.mnli.tokenizer import MNLITokenizer


def _model_factory(
    model_options: ModelOptions,
    sequence_options: SequenceOptions,
    tokenizer: MNLITokenizer,
    device_options: DeviceOptions,
    precision_options: PrecisionOptions,
) -> TypedModule[Tensor, Tensor, ModelForwardOutput]:
    model = LoopedMNLI(
        model_options, sequence_options, tokenizer.get_tokenizer_info()
    ).to(device=device_options.device, dtype=precision_options.dtype)
    return TypedModule(model)


def _loop_diagnostics_policy_factory(
    loop: LoopOptions,
    device: DeviceOptions,
    options: LoopDiagnosticsOptions,
    logger: StructuralLogger,
) -> LoopDiagnosticsPolicy:
    return LoopDiagnosticsPolicy(loop, device, options, logger)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the looped MNLI transformer")
    parser.add_argument(
        "--config-dir",
        default="conf",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
    )
    subparser = parser.add_subparsers(dest="command", required=True)

    train_parser = subparser.add_parser("train")
    train_parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
    )

    eval_parser = subparser.add_parser("eval")
    eval_parser.add_argument(
        "--data-file",
        type=Path,
        default=Path("data/mnli_matched.tsv"),
        help="MNLI TSV to evaluate (default: matched validation)",
    )
    eval_parser.add_argument("--batch-size", type=int, default=64)
    eval_parser.add_argument(
        "--checkpoint",
        required=True,
    )
    eval_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for batches.jsonl and summary.json (default: beside checkpoint)",
    )

    args = parser.parse_args()
    options = load_options(args.config_dir, args.override)
    container = build_container(options)

    if args.command == "train":
        trainer_factory = container[ModelTrainerFactory]

        make_model = container.partial(_model_factory)
        make_collator = container.partial(MNLICollator)

        trainer = trainer_factory.create(make_model(), make_collator())
        stage_registry = container[
            TrainingStageRegistry[MNLIBatch, ModelForwardOutput, Tensor]
        ]
        trainer.train(stage_registry.resolve("backbone_warmup"), args.resume)
    if args.command == "eval":
        data_options = MNLIDataOptions(
            seed=42,
            file_path=args.data_file,
            batch_size=args.batch_size,
            premise_column="sentence1",
            hypothesis_column="sentence2",
            label_column="gold_label",
            shuffle_cache_dir=Path("data/shuffled"),
            shuffle_chunk_size=10000,
        )
        evaluation_output_dir = args.output_dir or (
            Path(args.checkpoint).resolve().parent / "evaluation"
        )
        evaluation_options = EvaluationOptions(
            checkpoint=EvaluationCheckpointOptions(path=Path(args.checkpoint)),
            data=data_options,
            output=EvaluationOutputOptions(directory=evaluation_output_dir),
            diagnostics=LoopDiagnosticsOptions(),
        )
        evaluation_container = configure_evaluation(container, evaluation_options)
        evaluator_factory = evaluation_container[EvaluatorFactory]

        make_model = evaluation_container.partial(_model_factory)
        make_collator = evaluation_container.partial(MNLICollator)
        make_policy = evaluation_container.partial(_loop_diagnostics_policy_factory)
        make_data_provider = evaluation_container.partial(MNLIDataProvider)

        policy = make_policy()
        evaluator = evaluator_factory.create(
            make_model(), make_collator(), policy, make_data_provider()
        )
        evaluator.evaluate()

        summary = policy.collect_diagnostics()
        summary["checkpoint"] = str(Path(args.checkpoint).resolve())
        summary["data_file"] = str(args.data_file.resolve())
        (evaluation_output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(f"Evaluation logs written to {evaluation_output_dir}")


if __name__ == "__main__":
    main()
