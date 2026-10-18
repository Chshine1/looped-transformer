from pydantic import Field, model_validator

from looped_transformer.config.base import (
    ExclusiveProbability,
    FiniteFloat,
    OptionsModel,
    PositiveInt,
)
from looped_transformer.config.training import TrainingOptions


class TransformerOptions(OptionsModel):
    swi_glu_ffn_dimension: PositiveInt
    gelu_ffn_dimension: PositiveInt
    attention_heads_count: PositiveInt


class HaltingOptions(OptionsModel):
    residue_time_scaling: FiniteFloat = 0.75
    hazard_baseline_offset: FiniteFloat = -2.0
    hazard_baseline_scale: FiniteFloat = -2.0
    hazard_adjustment_scale: FiniteFloat = 0.9


class DropoutOptions(OptionsModel):
    token: ExclusiveProbability = 0.0
    embedding: ExclusiveProbability = 0.1
    attention: ExclusiveProbability = 0.1
    residual: ExclusiveProbability = 0.1
    ffn: ExclusiveProbability = 0.1
    classifier: ExclusiveProbability = 0.1
    controller: ExclusiveProbability = 0.1
    stochastic_depth: ExclusiveProbability = 0.0


class ModelOptions(OptionsModel):
    hidden_state_dimension: PositiveInt
    output_classes_count: PositiveInt
    max_iterations: PositiveInt
    max_sequence_length: PositiveInt
    dropout: DropoutOptions = Field(default_factory=DropoutOptions)

    transformer: TransformerOptions
    halting: HaltingOptions = Field(default_factory=HaltingOptions)
    training: TrainingOptions

    @model_validator(mode="after")
    def require_complete_attention_heads(self) -> "ModelOptions":
        heads = self.transformer.attention_heads_count
        if self.hidden_state_dimension % heads:
            raise ValueError(
                "hidden_state_dimension must be divisible by attention_heads_count"
            )
        return self
