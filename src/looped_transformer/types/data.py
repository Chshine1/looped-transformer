from dataclasses import dataclass
from enum import Enum


class DataSplit(Enum):
    TRAIN = "train"
    EVAL = "eval"


@dataclass(frozen=True, slots=True)
class MNLIBatch:
    premises: tuple[str, ...]
    hypotheses: tuple[str, ...]
    labels: tuple[str, ...]
