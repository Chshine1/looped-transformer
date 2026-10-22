from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from looped_transformer.config import DataOptions

PrimaryReduction = Literal["final_loop", "mean_across_loops"]


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    checkpoint_path: Path
    data: DataOptions
    output_dir: Path
    label_smoothing: float = 0.05
    primary_reduction: PrimaryReduction = "final_loop"
    shuffle_oracle_repeats: int = 20
