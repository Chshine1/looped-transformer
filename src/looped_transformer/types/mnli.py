from enum import Enum, unique


@unique
class LogicRelation(Enum):
    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2


def get_logic_relation(label: str) -> LogicRelation:
    label = label.lower().strip()
    relations = {
        "entailment": LogicRelation.ENTAILMENT,
        "contradiction": LogicRelation.CONTRADICTION,
        "neutral": LogicRelation.NEUTRAL,
    }
    try:
        return relations[label]
    except KeyError as error:
        raise ValueError(f"Unknown logic relation: {label}") from error
