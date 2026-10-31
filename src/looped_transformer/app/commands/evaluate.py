import json
from argparse import ArgumentParser, Namespace
from pathlib import Path

from looped_transformer.app.command import Command, CommandContext
from looped_transformer.app.commands.utils.modeling import create_model
from looped_transformer.app.container import configure_evaluation
from looped_transformer.core.config.runtime import DeviceOptions
from looped_transformer.core.evaluation.evaluator import EvaluationCheckpointOptions
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
                "(default: beside checkpoint)"
            ),
        )
        parser.add_argument("--summary-only", action="store_true")

    def execute(self, args: Namespace, context: CommandContext) -> None:
        stage_type = context.options.training.stage.get("type")
        if stage_type != "looped_clutrr":
            raise ValueError(f"Unknown training stage type: {stage_type!r}")
        output_dir = args.output_dir or (
            Path(args.checkpoint).resolve().parent / "evaluation"
        )
        diagnostics = LoopDiagnosticsOptions()

        data_file, policy = self._evaluate_clutrr(
            args, context, output_dir, diagnostics
        )

        self._write_summary(args, output_dir, data_file, policy)

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
    def _write_summary(
        args: Namespace,
        output_dir: Path,
        data_file: Path,
        policy: LoopDiagnosticsPolicy,
    ) -> None:
        summary = policy.collect_diagnostics()
        summary["checkpoint"] = str(Path(args.checkpoint).resolve())
        summary["data_file"] = str(data_file.resolve())
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        if args.summary_only:
            (output_dir / "batches.jsonl").unlink(missing_ok=True)
        print(f"Evaluation logs written to {output_dir}")
