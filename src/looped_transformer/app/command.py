from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import ClassVar

from lagom import Container

from looped_transformer.app.config import AppOptions


@dataclass(frozen=True)
class CommandContext:
    options: AppOptions
    container: Container


class Command(ABC):
    name: ClassVar[str]
    help: ClassVar[str]

    def configure_parser(self, parser: ArgumentParser) -> None:
        del parser

    @abstractmethod
    def execute(self, args: Namespace, context: CommandContext) -> None:
        pass
