from dataclasses import dataclass


@dataclass(frozen=True)
class TrajectoryReport:
    examples: int
    ce: list[float]
    accuracy: list[float]
    ce_change_standard_error: list[float]
    improved_fraction: list[float]
    worsened_fraction: list[float]
    prediction_flip_fraction: list[float]
    rescued_fraction: list[float]
    spoiled_fraction: list[float]
    best_ce_depth_counts: list[int]
    cost_optimal_depth_counts: list[int]
    any_depth_correct_accuracy: float
    oracle_ce: float
    token_relative_step: list[float]
    pooled_relative_step: list[float]
    normalized_relative_step: list[float]
    classifier_visible_step_energy_fraction: list[float]
    component_relative_updates: dict[str, list[float]]


@dataclass(frozen=True)
class GradientGroupReport:
    parameter_count: int
    per_depth_norm: list[float]
    cosine: list[list[float]]
    mean_objective_norm: float
    cancellation_ratio: float


@dataclass(frozen=True)
class GradientReport:
    examples: int
    groups: dict[str, GradientGroupReport]
    gamma_logit_per_depth: list[float]
    local_state_gradient_norm: list[float]
    future_state_gradient_norm: list[float]
    local_future_state_gradient_cosine: list[float]


@dataclass(frozen=True)
class ProbeReport:
    training_examples: int
    steps: int
    shared_ce: list[float]
    shared_accuracy: list[float]
    separate_ce: list[float]
    separate_accuracy: list[float]


@dataclass(frozen=True)
class CheckpointReport:
    checkpoint: str
    sha256: str
    epoch: int
    stage: str
    global_step: int
    recorded_best_accuracy: float
    gamma: float | None
    residual_branch_scales: list[float]
    injection_gate_rms: float
    evaluation: TrajectoryReport
    training_sample: TrajectoryReport
    gradients: GradientReport
    probes: ProbeReport
    interventions: dict[str, TrajectoryReport]
    residual_scaling_mode: str
    injection_scales: list[float]


@dataclass(frozen=True)
class WarmupReport:
    schema_version: int
    provenance: dict[str, object]
    checkpoints: list[CheckpointReport]
