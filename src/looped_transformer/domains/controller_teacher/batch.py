from dataclasses import dataclass

from torch import Tensor

from looped_transformer.domains.mnli.types.batch import MNLIBatch


@dataclass(frozen=True, slots=True)
class ControllerTeacherBatch:
    """One MNLI batch and the frozen backbone trajectory produced for it."""

    mnli: MNLIBatch
    timed_hidden_states: Tensor
    padding_mask: Tensor
    timed_logits: Tensor
