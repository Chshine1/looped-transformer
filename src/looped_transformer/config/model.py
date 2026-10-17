import math
from dataclasses import dataclass
from typing import Any, Self

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
class DropoutOptions:
    token: float = 0.0
    embedding: float = 0.1
    attention: float = 0.1
    residual: float = 0.1
    ffn: float = 0.1
    classifier: float = 0.1
    controller: float = 0.1
    stochastic_depth: float = 0.0

    def __post_init__(self) -> None:
        for name, probability in (
            ("token", self.token),
            ("embedding", self.embedding),
            ("attention", self.attention),
            ("residual", self.residual),
            ("ffn", self.ffn),
            ("classifier", self.classifier),
            ("controller", self.controller),
            ("stochastic_depth", self.stochastic_depth),
        ):
            if not isinstance(probability, int | float) or not math.isfinite(
                probability
            ):
                raise TypeError(f"{name} dropout must be a finite number")
            if not 0 <= probability < 1:
                raise ValueError(f"{name} dropout must be in [0, 1)")

    @classmethod
    def model_validate(cls, value: Any) -> Self:
        if isinstance(value, int | float) and not isinstance(value, bool):
            # Preserve the historical scalar behavior. Embedding and classifier
            # dropout modules existed only nominally or not at all.
            return cls(
                token=0.0,
                embedding=0.0,
                attention=value,
                residual=value,
                ffn=value,
                classifier=0.0,
                controller=value,
                stochastic_depth=0.0,
            )
        if not isinstance(value, dict):
            raise TypeError("dropout must be a number or mapping")
        unexpected = value.keys() - {
            "embedding",
            "token",
            "attention",
            "residual",
            "ffn",
            "classifier",
            "controller",
            "stochastic_depth",
        }
        if unexpected:
            raise ValueError(
                "Unexpected dropout options: " + ", ".join(sorted(unexpected))
            )
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ModelOptions:
    hidden_state_dimension: int
    output_classes_count: int
    max_iterations: int
    max_sequence_length: int
    dropout: DropoutOptions

    transformer: TransformerOptions
    halting: HaltingOptions
    training: TrainingOptions
    evaluation: EvaluationOptions
