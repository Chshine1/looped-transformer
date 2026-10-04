from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformerOptions:
    ffn_dimension: int
    attention_heads_count: int


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    dropout: float = 0.1
    residue_time_scaling: float = 0.75

    halt_hazard_baseline_offset: float = -2.0
    halt_hazard_baseline_scale: float = -2.0
    halt_hazard_adjustment_scale: float = 0.9


@dataclass(frozen=True, slots=True)
class EvaluationOptions:
    halt_hazard_threshold: float = 0.6
    halt_survival_threshold: float = 0.05


@dataclass(frozen=True, slots=True)
class ModelOptions:
    hidden_state_dimension: int
    output_classes_count: int
    max_iterations: int
    max_sequence_length: int

    transformer: TransformerOptions
    training: TrainingOptions
    evaluation: EvaluationOptions
