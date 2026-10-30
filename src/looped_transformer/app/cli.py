import argparse

from looped_transformer.app.command import CommandContext
from looped_transformer.app.commands.evaluate import EvaluateCommand
from looped_transformer.app.commands.train import TrainCommand
from looped_transformer.app.config import load_options
from looped_transformer.app.container import build_container
from looped_transformer.app.registry import CommandRegistry


def create_command_registry() -> CommandRegistry:
    registry = CommandRegistry()
    registry.register(TrainCommand())
    registry.register(EvaluateCommand())
    return registry


def main() -> None:
    registry = create_command_registry()
    parser = argparse.ArgumentParser(description="Train the looped text classifier")
    parser.add_argument("--config-dir", default="conf")
    parser.add_argument("--config-name", default="config")
    parser.add_argument("--override", action="append", default=[])
    registry.configure_parser(parser)

    args = parser.parse_args()
    options = load_options(args.config_dir, args.override, config_name=args.config_name)
    registry.execute(
        args.command,
        args,
        CommandContext(options=options, container=build_container(options)),
    )


if __name__ == "__main__":
    main()
