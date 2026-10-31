from dataclasses import dataclass

from torch import Tensor

from looped_transformer.domains.clutrr.types import CLUTRRBatch


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    source: CLUTRRBatch
    timed_hidden_states: Tensor
    padding_mask: Tensor
    final_logits: Tensor
