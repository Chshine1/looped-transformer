from enum import Enum, unique


@unique
class LogicRelation(Enum):
    ENTAILMENT = 0
    CONTRADICTION = 1
    NEUTRAL = 2
