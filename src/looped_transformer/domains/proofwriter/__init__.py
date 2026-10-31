"""ProofWriter auxiliary-task integration."""

from .data_source import (
    PROOFWRITER_TASK,
    ProofWriterAuxiliaryOptions,
    ProofWriterAuxiliarySource,
)
from .types import (
    PROOFWRITER_LABELS,
    ProofWriterBatch,
    proofwriter_label_index,
)

__all__ = [
    "PROOFWRITER_LABELS",
    "PROOFWRITER_TASK",
    "ProofWriterAuxiliaryOptions",
    "ProofWriterAuxiliarySource",
    "ProofWriterBatch",
    "proofwriter_label_index",
]
