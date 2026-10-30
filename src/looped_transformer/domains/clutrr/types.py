from dataclasses import dataclass

# Stable project ordering for the gendered relations emitted by CLUTRR v1.3.
# Keep this independent of the order in any generated split.
CLUTRR_RELATIONS = (
    "aunt",
    "brother",
    "daughter",
    "daughter-in-law",
    "father",
    "father-in-law",
    "granddaughter",
    "grandfather",
    "grandmother",
    "grandson",
    "mother",
    "mother-in-law",
    "nephew",
    "niece",
    "sister",
    "son",
    "son-in-law",
    "uncle",
)

_RELATION_TO_INDEX = {
    relation: index for index, relation in enumerate(CLUTRR_RELATIONS)
}


def relation_index(label: str) -> int:
    try:
        return _RELATION_TO_INDEX[label.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unknown CLUTRR relation: {label!r}") from error


@dataclass(frozen=True, slots=True)
class CLUTRRBatch:
    ids: tuple[str, ...]
    stories: tuple[str, ...]
    queries: tuple[str, ...]
    labels: tuple[str, ...]
    depths: tuple[int, ...]
