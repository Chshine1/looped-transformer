from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TransformerOptions:
    swi_glu_ffn_dimension: int
    gelu_ffn_dimension: int
    attention_heads_count: int


@dataclass(frozen=True, slots=True)
class TrainingOptions:
    max_epochs: int
    learning_rate: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_epsilon: float
    max_gradient_norm: float
    warmup_ratio: float
    warmup_start_factor: float
    minimum_learning_rate: float
    expected_halting_time: int
    halting_loss_scale: float
    auxiliary_loss_scale: float
    dropout: float
    residue_time_scaling: float
    gradient_accumulation_steps: int = 1
    log_every_steps: int = 50
    save_every_epochs: int = 1

    halt_hazard_baseline_offset: float = -2.0
    halt_hazard_baseline_scale: float = -2.0
    halt_hazard_adjustment_scale: float = 0.9


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

    transformer: TransformerOptions
    training: TrainingOptions
    evaluation: EvaluationOptions
