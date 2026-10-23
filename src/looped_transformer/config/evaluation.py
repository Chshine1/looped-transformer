from dataclasses import dataclass
from pathlib import Path

from looped_transformer.config import DataOptions


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    checkpoint_path: Path
    data: DataOptions
    output_dir: Path
    label_smoothing: float = 0.05
    shuffle_oracle_repeats: int = 20
