from enum import Enum, unique


@unique
class LogicRelation(Enum):
    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2


def get_logic_relation(label: str) -> LogicRelation:
    label = label.lower().strip()

    if label == "entailment":
        return LogicRelation.ENTAILMENT
    elif label == "contradiction":
        return LogicRelation.CONTRADICTION
    elif label == "neutral":
        return LogicRelation.NEUTRAL

    raise ValueError(f"Unknown logic relation: {label}")
