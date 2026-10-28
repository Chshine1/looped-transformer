import argparse
import json
from pathlib import Path

from torch import Tensor

from looped_transformer.app.config import load_options
from looped_transformer.app.container import build_container, configure_evaluation
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.core.evaluation.evaluator import (
    EvaluationCheckpointOptions,
)
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.mnli.config.options import LoopOptions
from looped_transformer.domains.mnli.evaluation.evaluation import EvaluationOptions
from looped_transformer.domains.mnli.evaluation.loop_diagnostics import (
    LoopDiagnosticsOptions,
    LoopDiagnosticsPolicy,
)
from looped_transformer.domains.mnli.modules.anchored_sampled_ce_loss import (
    BackboneLossInput,
)
from looped_transformer.domains.mnli.modules.model import LoopedMNLI, ModelOptions
from looped_transformer.domains.mnli.services.data_provider import (
    MNLIDataOptions,
    MNLIDataProvider,
)
from looped_transformer.domains.mnli.services.tokenizer import MNLITokenizer
from looped_transformer.domains.mnli.training.backbone_collator import (
    BackboneTrainingCollator,
)
from looped_transformer.domains.mnli.training.collator import MNLICollator
from looped_transformer.domains.mnli.training.pretrained_embedding_initializer import (
    initialize_pretrained_input_embeddings,
)
from looped_transformer.domains.mnli.training.stage import (
    LoopedMNLIStageTemplate,
)
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.mnli.types.model import ModelForwardOutput


def _model_factory(
    model_options: ModelOptions,
    sequence_options: SequenceOptions,
    tokenizer: MNLITokenizer,
    device_options: DeviceOptions,
    precision_options: PrecisionOptions,
) -> TypedModule[Tensor, Tensor, Tensor, ModelForwardOutput]:
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

    subparser.add_parser("train")

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
    eval_parser.add_argument("--summary-only", action="store_true")

    args = parser.parse_args()
    options = load_options(args.config_dir, args.override)
    container = build_container(options)

    if args.command == "train":
        trainer_factory = container[ModelTrainerFactory]
        stage_type = options.training.stage.get("type")
        if stage_type != "looped_mnli":
            raise ValueError(f"Unknown training stage type: {stage_type!r}")
        template = LoopedMNLIStageTemplate.model_validate(
            {
                key: value
                for key, value in options.training.stage.items()
                if key != "type"
            }
        )
        make_model = container.magic_partial(_model_factory)
        device_options = container[DeviceOptions]
        options.runtime.reproducibility.activate(device_options.device)
        tokenizer = container[MNLITokenizer]
        backbone = make_model()
        if options.training.resume_from is None:
            initialization = options.model.embedding.initialization
            if (
                options.tokenizer.source.pretrained_model
                != initialization.pretrained_model
                or options.tokenizer.source.revision != initialization.revision
            ):
                raise ValueError(
                    "tokenizer and pretrained input embeddings must use the same "
                    "model and revision"
                )
            if not isinstance(backbone.torch, LoopedMNLI):
                raise TypeError("Expected a LoopedMNLI backbone")
            initialize_pretrained_input_embeddings(
                backbone.torch,
                tokenizer.get_tokenizer_info(),
                initialization,
                options.model.embedding.segment_types_count,
                options.runtime.reproducibility.seed,
            )
        collator = BackboneTrainingCollator(
            tokenizer,
            template.teacher,
            template.loss_options.distillation_scale,
            device_options.device,
        )
        stage_registry = container[
            TrainingStageRegistry[MNLIBatch, ModelForwardOutput, BackboneLossInput]
        ]
        trainer = trainer_factory.create(backbone, collator, stage_registry.resolve())
        trainer.train()
    elif args.command == "eval":
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

        make_model = evaluation_container.magic_partial(_model_factory)
        make_collator = evaluation_container.magic_partial(MNLICollator)
        make_policy = evaluation_container.magic_partial(
            _loop_diagnostics_policy_factory
        )
        make_data_provider = evaluation_container.magic_partial(MNLIDataProvider)

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
        if args.summary_only:
            (evaluation_output_dir / "batches.jsonl").unlink()
        print(f"Evaluation logs written to {evaluation_output_dir}")


if __name__ == "__main__":
    main()
