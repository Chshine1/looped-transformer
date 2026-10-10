import argparse

from looped_transformer.checkpoint_manager import (
    CheckpointManagerFactory,
)
from looped_transformer.config import load_options
from looped_transformer.data_reader import MNLIDataReader
from looped_transformer.runner import MNLIRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the looped MNLI transformer")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--resume",
        metavar="CHECKPOINT",
        help="resume from a checkpoint path, or use 'latest'",
    )
    args = parser.parse_args()
    options = load_options(args.config)
    checkpoint_manager_factory = CheckpointManagerFactory(options.model, options.data)
    data_reader = MNLIDataReader(options.runtime, options.data)
    MNLIRunner(
        checkpoint_manager_factory,
        data_reader,
        options.model,
        options.data,
        options.tokenizer,
        options.runtime,
    ).train(args.resume)


if __name__ == "__main__":
    main()
