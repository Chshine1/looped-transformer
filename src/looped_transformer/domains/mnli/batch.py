"""MNLI batch data transfer object."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MNLIBatch:
    premises: tuple[str, ...]
    hypotheses: tuple[str, ...]
    labels: tuple[str, ...]
