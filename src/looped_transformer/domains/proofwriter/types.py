from dataclasses import dataclass

PROOFWRITER_LABELS = ("false", "true", "unknown")

_LABEL_TO_INDEX = {label: index for index, label in enumerate(PROOFWRITER_LABELS)}


def proofwriter_label_index(label: str) -> int:
    try:
        return _LABEL_TO_INDEX[label.strip().lower()]
    except KeyError as error:
        raise ValueError(f"Unknown ProofWriter label: {label!r}") from error


@dataclass(frozen=True, slots=True)
class ProofWriterBatch:
    ids: tuple[str, ...]
    theories: tuple[str, ...]
    queries: tuple[str, ...]
    labels: tuple[str, ...]
    # Unknown examples are unprovable and therefore have no proof depth.
    depths: tuple[int | None, ...]

    def __post_init__(self) -> None:
        sizes = {
            len(self.ids),
            len(self.theories),
            len(self.queries),
            len(self.labels),
            len(self.depths),
        }
        if len(sizes) != 1:
            raise ValueError("ProofWriter batch fields must have equal lengths")
        if not self.ids:
            raise ValueError("ProofWriter batch must not be empty")
