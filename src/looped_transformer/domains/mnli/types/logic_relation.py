from enum import Enum, unique
from typing import Self


@unique
class LogicRelation(Enum):
    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2

    @classmethod
    def from_string(cls, label: str) -> Self:
        label = label.lower().strip()
        relations = {
            "entailment": cls.ENTAILMENT,
            "contradiction": cls.CONTRADICTION,
            "neutral": cls.NEUTRAL,
        }
        try:
            return relations[label]
        except KeyError as error:
            raise ValueError(f"Unknown logic relation: {label}") from error
