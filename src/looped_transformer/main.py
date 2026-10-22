import argparse
import json
from pathlib import Path

from looped_transformer.checkpoint_manager import CheckpointManagerFactory
from looped_transformer.config import DataOptions, load_options
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.evaluation import Evaluator, StructuredEvaluationLogger
from looped_transformer.evaluation.policies import LoopDiagnosticsPolicy
from looped_transformer.runner import MNLIRunner
from looped_transformer.stages import build_stage_templates
from looped_transformer.tokenizer import MNLITokenizer


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
    tokenizer = MNLITokenizer(options.model.max_sequence_length)

    if args.command == "train":
        checkpoint_manager_factory = CheckpointManagerFactory(
            options.model,
            options.data,
            options.runtime,
        )
        stages = build_stage_templates(options.model.training)
        runner = MNLIRunner(
            tokenizer,
            checkpoint_manager_factory,
            options.model,
            options.runtime,
        )
        runner.train(stages, args.resume)
    if args.command == "eval":
        data_options = DataOptions(
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
        policy = LoopDiagnosticsPolicy(
            max_iterations=options.model.max_iterations,
            device=options.runtime.device,
            label_smoothing=0.05,
            primary_reduction="final_loop",
            shuffle_oracle_repeats=20,
            logger=StructuredEvaluationLogger(evaluation_output_dir),
        )
        evaluator = Evaluator(
            args.checkpoint,
            MNLIDataReader(data_options),
            tokenizer,
            options.model,
            options.runtime,
            policy,
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
