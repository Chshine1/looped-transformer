import argparse
from pathlib import Path

from looped_transformer.checkpoint_manager import CheckpointManagerFactory
from looped_transformer.config import DataOptions
from looped_transformer.config import load_options
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.evaluator import Evaluator
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
        "--checkpoint",
        required=True,
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
            file_path=Path("data/mnli_train.tsv"),
            batch_size=64,
            premise_column="sentence1",
            hypothesis_column="sentence2",
            label_column="gold_label",
            shuffle_cache_dir=Path("data/shuffled"),
            shuffle_chunk_size=10000,
        )
        evaluator = Evaluator(
            args.checkpoint,
            MNLIDataReader(data_options),
            options.model,
            options.runtime,
        )
        result = evaluator.evaluate()
        print(result)


if __name__ == "__main__":
    main()
