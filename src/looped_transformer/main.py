import argparse

from looped_transformer.checkpoint_manager import CheckpointManagerFactory
from looped_transformer.config import load_options
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

    train_parser = subparser.add_parser("train", help="train the model")
    train_parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="resume from a checkpoint path, or use 'latest'",
    )

    args = parser.parse_args()
    options = load_options(args.config_dir, args.override)
    tokenizer = MNLITokenizer(options.model.max_sequence_length)
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

    if args.command == "train":
        runner.train(stages, args.resume)


if __name__ == "__main__":
    main()
