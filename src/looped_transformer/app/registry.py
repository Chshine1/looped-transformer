from argparse import ArgumentParser, Namespace

from looped_transformer.app.command import Command, CommandContext


class CommandRegistry:
    def __init__(self) -> None:
        self._commands: dict[str, Command] = {}

    def register(self, command: Command) -> None:
        if command.name in self._commands:
            raise ValueError(f"Command {command.name!r} is already registered")
        self._commands[command.name] = command

    def configure_parser(self, parser: ArgumentParser) -> None:
        subparsers = parser.add_subparsers(dest="command", required=True)
        for command in self._commands.values():
            command_parser = subparsers.add_parser(command.name, help=command.help)
            command.configure_parser(command_parser)

    def execute(
        self, command_name: str, args: Namespace, context: CommandContext
    ) -> None:
        try:
            command = self._commands[command_name]
        except KeyError as error:
            raise ValueError(f"Unknown command: {command_name!r}") from error
        command.execute(args, context)
