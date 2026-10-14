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
