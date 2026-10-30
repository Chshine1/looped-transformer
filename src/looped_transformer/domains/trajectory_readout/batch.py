from dataclasses import dataclass

from torch import Tensor

from looped_transformer.domains.mnli.types.batch import MNLIBatch


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    mnli: MNLIBatch
    timed_hidden_states: Tensor
    padding_mask: Tensor
    final_logits: Tensor
