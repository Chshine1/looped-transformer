import json
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any

from looped_transformer.app.command import Command, CommandContext
from looped_transformer.app.commands.utils.modeling import create_model
from looped_transformer.app.container import configure_evaluation
from looped_transformer.core.abstractions.typed_module import TypedBranchedModule
from looped_transformer.core.checkpointing import CheckpointLoader
from looped_transformer.core.config.runtime import DeviceOptions
from looped_transformer.core.evaluation.evaluator import (
    EvaluationCheckpointOptions,
    Evaluator,
)
from looped_transformer.core.evaluation.evaluator_factory import EvaluatorFactory
from looped_transformer.core.logging import StructuralLogger
from looped_transformer.core.logging.structural_logger import EvaluationOutputOptions
from looped_transformer.domains.clutrr.collator import (
    CLUTRRCollator,
    CLUTRRCollatorOptions,
)
from looped_transformer.domains.clutrr.data_provider import (
    CLUTRRDataOptions,
    CLUTRRDataProvider,
)
from looped_transformer.domains.clutrr.types import CLUTRR_RELATIONS, relation_index
from looped_transformer.domains.text_classification.loop_diagnostics import (
    LoopDiagnosticsOptions,
    LoopDiagnosticsPolicy,
)
from looped_transformer.domains.text_classification.tokenizer import PairTokenizer
from looped_transformer.domains.trajectory_readout import (
    TrajectoryCollectionPolicy,
    TrajectoryDataProvider,
    TrajectoryReadout,
    TrajectoryReadoutCollator,
    TrajectoryReadoutEvaluationPolicy,
    TrajectoryReadoutStageTemplate,
)


class EvaluateCommand(Command):
    name = "eval"
    help = "Evaluate a checkpoint"

    def configure_parser(self, parser: ArgumentParser) -> None:
        parser.add_argument(
            "--data-file",
            type=Path,
            help="CLUTRR evaluation split (default: data/clutrr-v1.3/valid.jsonl)",
        )
        parser.add_argument(
            "--data-manifest",
            type=Path,
            help="CLUTRR manifest (default: manifest.json beside --data-file)",
        )
        parser.add_argument("--batch-size", type=int, default=64)
        parser.add_argument("--checkpoint", required=True)
        parser.add_argument(
            "--output-dir",
            type=Path,
            help=(
                "Directory for batches.jsonl and summary.json "
                "(default: beside checkpoints/)"
            ),
        )
        parser.add_argument("--summary-only", action="store_true")

    def execute(self, args: Namespace, context: CommandContext) -> None:
        stage_type = context.options.training.stage.get("type")
        if stage_type not in {"looped_clutrr", "trajectory_readout"}:
            raise ValueError(f"Unknown training stage type: {stage_type!r}")
        output_dir = args.output_dir or (
            Path(args.checkpoint).resolve().parent.parent / "evaluation"
        )
        if stage_type == "looped_clutrr":
            data_file, policy = self._evaluate_clutrr(
                args, context, output_dir, LoopDiagnosticsOptions()
            )
        else:
            data_file, policy = self._evaluate_trajectory_readout(
                args, context, output_dir
            )

        self._write_summary(args, context, output_dir, data_file, policy)

    @staticmethod
    def _evaluate_clutrr(
        args: Namespace,
        context: CommandContext,
        output_dir: Path,
        diagnostics: LoopDiagnosticsOptions,
    ) -> tuple[Path, LoopDiagnosticsPolicy]:
        data_file = args.data_file or Path("data/clutrr-v1.3/valid.jsonl")
        data_options = CLUTRRDataOptions(
            seed=42,
            file_path=data_file,
            manifest_path=args.data_manifest or data_file.parent / "manifest.json",
            batch_size=args.batch_size,
        )
        container = configure_evaluation(
            context.container,
            EvaluationCheckpointOptions(path=Path(args.checkpoint)),
            EvaluationOutputOptions(directory=output_dir),
            diagnostics,
            data_options,
        )
        policy = LoopDiagnosticsPolicy(
            context.options.model.loop,
            container[DeviceOptions],
            diagnostics,
            container[StructuralLogger],
            label_to_index=relation_index,
            classes_count=len(CLUTRR_RELATIONS),
        )
        evaluator = container[EvaluatorFactory].create(
            container.magic_partial(create_model)(),
            CLUTRRCollator(
                container[PairTokenizer],
                CLUTRRCollatorOptions(main_loss_scale=1.0),
            ),
            policy,
            CLUTRRDataProvider(data_options),
        )
        evaluator.evaluate()
        return data_file, policy

    @staticmethod
    def _evaluate_trajectory_readout(
        args: Namespace,
        context: CommandContext,
        output_dir: Path,
    ) -> tuple[Path, TrajectoryReadoutEvaluationPolicy]:
        raw_stage = context.options.training.stage
        template = TrajectoryReadoutStageTemplate.model_validate(
            {key: value for key, value in raw_stage.items() if key != "type"}
        )
        data_file = args.data_file or Path("data/clutrr-v1.3/valid.jsonl")
        data_options = CLUTRRDataOptions(
            seed=42,
            file_path=data_file,
            manifest_path=args.data_manifest or data_file.parent / "manifest.json",
            batch_size=args.batch_size,
        )
        source = CLUTRRDataProvider(data_options)
        tokenizer = context.container[PairTokenizer]
        backbone = context.container.magic_partial(create_model)()
        backbone_collator = CLUTRRCollator(
            tokenizer,
            CLUTRRCollatorOptions(main_loss_scale=1.0),
        )
        collection_policy = TrajectoryCollectionPolicy(
            context.options.model.loop.max_iterations
        )
        backbone_evaluator = Evaluator(
            backbone,
            backbone_collator,
            collection_policy,
            source,
            CheckpointLoader(context.options.runtime.device),
            context.options.runtime.device,
            context.options.runtime.precision,
            EvaluationCheckpointOptions(path=template.backbone_checkpoint),
        )
        trajectories = TrajectoryDataProvider(
            source=source,
            evaluator=backbone_evaluator,
            collection_policy=collection_policy,
            output_batch_size=args.batch_size,
        )
        readout = TypedBranchedModule(
            TrajectoryReadout(
                state_dimension=context.options.model.hidden_state_dimension,
                classes_count=len(CLUTRR_RELATIONS),
                options=template.model_options,
            ).to(
                device=context.options.runtime.device.device,
                dtype=context.options.runtime.precision.dtype,
            )
        )
        policy = TrajectoryReadoutEvaluationPolicy(
            label_smoothing=template.loss_options.label_smoothing,
            logger=StructuralLogger(EvaluationOutputOptions(directory=output_dir)),
        )
        evaluator = Evaluator(
            readout,
            TrajectoryReadoutCollator(),
            policy,
            trajectories,
            CheckpointLoader(context.options.runtime.device),
            context.options.runtime.device,
            context.options.runtime.precision,
            EvaluationCheckpointOptions(path=Path(args.checkpoint)),
        )
        evaluator.evaluate()
        return data_file, policy

    @staticmethod
    def _write_summary(
        args: Namespace,
        context: CommandContext,
        output_dir: Path,
        data_file: Path,
        policy: Any,
    ) -> None:
        summary = policy.collect_diagnostics()
        summary["checkpoint"] = str(Path(args.checkpoint).resolve())
        summary["data_file"] = str(data_file.resolve())
        stage_type = context.options.training.stage.get("type")
        summary["stage_type"] = stage_type
        if stage_type == "trajectory_readout":
            raw_backbone_checkpoint = context.options.training.stage.get(
                "backbone_checkpoint"
            )
            summary["backbone_checkpoint"] = str(
                Path(str(raw_backbone_checkpoint)).resolve()
            )
            raw_model_options = context.options.training.stage.get("model_options")
            if isinstance(raw_model_options, dict):
                summary["readout_architecture"] = raw_model_options.get("architecture")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if args.summary_only:
            (output_dir / "batches.jsonl").unlink(missing_ok=True)
        print(f"Evaluation logs written to {output_dir}")
