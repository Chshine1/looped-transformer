from looped_transformer.core.config.base import (
    OptionsModel,
    PositiveFloat,
    PositiveInt,
)


class LoopOptions(OptionsModel):
    max_iterations: PositiveInt
    time_rope_base_frequency: PositiveFloat
