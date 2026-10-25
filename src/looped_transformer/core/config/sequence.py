"""Shared sequence-shape options."""

from looped_transformer.core.config.base import OptionsModel, PositiveInt


class SequenceOptions(OptionsModel):
    max_sequence_length: PositiveInt
