import math
from dataclasses import dataclass, field
from typing import Literal

from looped_transformer.config.training import TrainingOptions


@dataclass(frozen=True, slots=True)
class TransformerOptions:
    swi_glu_ffn_dimension: int
    gelu_ffn_dimension: int
    attention_heads_count: int


@dataclass(frozen=True, slots=True)
class HaltingOptions:
    residue_time_scaling: float = 0.75
    hazard_baseline_offset: float = -2.0
    hazard_baseline_scale: float = -2.0
    hazard_adjustment_scale: float = 0.9


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    halt_hazard_threshold: float = 0.6
    halt_survival_threshold: float = 0.05
    batch_size: int = 64
    evaluate_every_epochs: int = 1


@dataclass(frozen=True, slots=True)
class ResidualScalingOptions:
    # Missing configuration preserves the behavior of historical checkpoints.
    mode: Literal["geometric", "static"] = "geometric"
    expected_iterations: float | None = None
    multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in ("geometric", "static"):
            raise ValueError("residual scaling mode must be geometric or static")
        if not math.isfinite(self.multiplier) or self.multiplier <= 0:
            raise ValueError("residual multiplier must be finite and positive")
        if self.expected_iterations is not None and (
            not math.isfinite(self.expected_iterations) or self.expected_iterations < 1
        ):
            raise ValueError("expected_iterations must be finite and at least one")
        if self.mode == "geometric" and (
            self.expected_iterations is not None or self.multiplier != 1.0
        ):
            raise ValueError(
                "expected_iterations and multiplier apply only to static scaling"
            )


@dataclass(frozen=True, slots=True)
class ModelOptions:
    hidden_state_dimension: int
    output_classes_count: int
    max_iterations: int
    max_sequence_length: int
    dropout: float

    transformer: TransformerOptions
    halting: HaltingOptions
    training: TrainingOptions
    evaluation: EvaluationOptions
    residual_scaling: ResidualScalingOptions = field(
        default_factory=ResidualScalingOptions
    )
