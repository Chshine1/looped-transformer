import argparse
import json
from pathlib import Path

from torch import Tensor

from looped_transformer.app.config import load_options
from looped_transformer.app.container import build_container, configure_evaluation
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.checkpointing import CheckpointLoader
from looped_transformer.core.config.runtime import DeviceOptions, PrecisionOptions
from looped_transformer.core.config.sequence import SequenceOptions
from looped_transformer.core.evaluation.evaluator import (
    EvaluationCheckpointOptions,
    Evaluator,
)
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.controller_teacher import (
    ControllerTeacher,
    ControllerTeacherBatch,
    ControllerTeacherCollator,
    ControllerTeacherCollectionPolicy,
    ControllerTeacherDataProvider,
    ControllerTeacherLossInput,
    ControllerTeacherOptions,
    ControllerTeacherStageTemplate,
    register_controller_teacher_stage_template,
)
from looped_transformer.domains.guided_backbone import (
    ControllerGuidedBackbone,
    ControllerGuidedBackboneStageTemplate,
    GuidedBackboneOutput,
    register_controller_guided_backbone_stage,
)
from looped_transformer.domains.mnli.config.options import LoopOptions
from looped_transformer.domains.mnli.evaluation.evaluation import EvaluationOptions
from looped_transformer.domains.mnli.evaluation.loop_diagnostics import (
    LoopDiagnosticsOptions,
    LoopDiagnosticsPolicy,
)
from looped_transformer.domains.mnli.modules.model import LoopedMNLI, ModelOptions
from looped_transformer.domains.mnli.services.data_provider import (
    MNLIDataOptions,
    MNLIDataProvider,
)
from looped_transformer.domains.mnli.services.tokenizer import MNLITokenizer
from looped_transformer.domains.mnli.training.collator import MNLICollator
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.mnli.types.model import ModelForwardOutput


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

    args = parser.parse_args()
    options = load_options(args.config_dir, args.override)
    container = build_container(options)

    if args.command == "train":
        trainer_factory = container[ModelTrainerFactory]
        stage_type = options.training.stage.get("type")
        if stage_type == "looped_mnli":
            make_model = container.magic_partial(_model_factory)
            make_collator = container.magic_partial(MNLICollator)
            stage_registry = container[
                TrainingStageRegistry[MNLIBatch, ModelForwardOutput, Tensor]
            ]
            trainer = trainer_factory.create(
                make_model(), make_collator(), stage_registry.resolve()
            )
        elif stage_type == "controller_teacher":
            device_options = container[DeviceOptions]
            precision_options = container[PrecisionOptions]
            sequence_options = container[SequenceOptions]
            tokenizer = container[MNLITokenizer]

            teacher_registry = TrainingStageRegistry[
                ControllerTeacherBatch,
                Tensor,
                ControllerTeacherLossInput,
            ](options.training)

            def create_teacher_data_provider(
                template: ControllerTeacherStageTemplate,
            ) -> ControllerTeacherDataProvider:
                source = MNLIDataProvider(template.data_options)
                collector = ControllerTeacherCollectionPolicy(
                    options.model.loop.max_iterations
                )
                backbone = _model_factory(
                    options.model,
                    sequence_options,
                    tokenizer,
                    device_options,
                    precision_options,
                )
                evaluator = Evaluator(
                    backbone,
                    MNLICollator(tokenizer),
                    collector,
                    source,
                    CheckpointLoader(device_options),
                    device_options,
                    precision_options,
                    EvaluationCheckpointOptions(path=template.source_checkpoint),
                )
                return ControllerTeacherDataProvider(
                    source,
                    evaluator,
                    collector,
                    template.training_batch_size,
                )

            register_controller_teacher_stage_template(
                teacher_registry,
                create_teacher_data_provider,
            )
            teacher_options = container[ControllerTeacherOptions]
            teacher = TypedModule(
                ControllerTeacher(teacher_options, sequence_options).to(
                    device=device_options.device,
                    dtype=precision_options.dtype,
                )
            )
            trainer = trainer_factory.create(
                teacher,
                ControllerTeacherCollator(),
                teacher_registry.resolve(),
            )
        elif stage_type == "controller_guided_backbone":
            device_options = container[DeviceOptions]
            precision_options = container[PrecisionOptions]
            sequence_options = container[SequenceOptions]
            tokenizer = container[MNLITokenizer]
            template = ControllerGuidedBackboneStageTemplate.model_validate(
                {
                    key: value
                    for key, value in options.training.stage.items()
                    if key != "type"
                }
            )

            backbone_typed = _model_factory(
                options.model,
                sequence_options,
                tokenizer,
                device_options,
                precision_options,
            )
            backbone = backbone_typed.torch
            if not isinstance(backbone, LoopedMNLI):
                raise TypeError("Expected a LoopedMNLI backbone")
            backbone_loader = CheckpointLoader(device_options)
            try:
                backbone_loader.load(template.source_checkpoint)
                backbone_loader.load_model(backbone)
            finally:
                backbone_loader.dispose()

            teacher_options = container[ControllerTeacherOptions]
            controller = ControllerTeacher(teacher_options, sequence_options).to(
                device=device_options.device,
                dtype=precision_options.dtype,
            )
            controller_loader = CheckpointLoader(device_options)
            try:
                controller_loader.load(template.controller_checkpoint)
                controller_loader.load_model(controller)
            finally:
                controller_loader.dispose()

            guided_registry = TrainingStageRegistry[
                MNLIBatch,
                GuidedBackboneOutput,
                Tensor,
            ](options.training)
            register_controller_guided_backbone_stage(guided_registry)
            guided_model = TypedModule(
                ControllerGuidedBackbone(
                    backbone,
                    controller,
                    template.guidance_options,
                )
            )
            trainer = trainer_factory.create(
                guided_model,
                MNLICollator(tokenizer),
                guided_registry.resolve(),
            )
        else:
            raise ValueError(f"Unknown training stage type: {stage_type!r}")
        trainer.train()
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
        print(f"Evaluation logs written to {evaluation_output_dir}")


if __name__ == "__main__":
    main()
