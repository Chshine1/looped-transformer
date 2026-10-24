from looped_transformer.config.base import (
    ExclusiveProbability,
    FiniteFloat,
    OptionsModel,
    PositiveFloat,
    PositiveInt,
)


class TransformerOptions(OptionsModel):
    swi_glu_ffn_dimension: PositiveInt
    gelu_ffn_dimension: PositiveInt
    attention_heads_count: PositiveInt


class LoopOptions(OptionsModel):
    max_iterations: PositiveInt
    time_rope_base_frequency: PositiveFloat = 4.8


class HaltingOptions(OptionsModel):
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
