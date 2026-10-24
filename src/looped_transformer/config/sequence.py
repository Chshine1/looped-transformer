from looped_transformer.config.base import OptionsModel, PositiveInt


class SequenceOptions(OptionsModel):
    max_sequence_length: PositiveInt
