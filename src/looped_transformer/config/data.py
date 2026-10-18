from looped_transformer.config.base import NonNegativeInt
from looped_transformer.config.base import (
    ConfigPath,
    NonEmptyString,
    OptionsModel,
    PositiveInt,
)


class CommonDataOptions(OptionsModel):
    output_dir: ConfigPath


class DataOptions(OptionsModel):
    seed: NonNegativeInt

    file_path: ConfigPath

    batch_size: PositiveInt
    premise_column: NonEmptyString
    hypothesis_column: NonEmptyString
    label_column: NonEmptyString

    shuffle_cache_dir: ConfigPath
    shuffle_chunk_size: PositiveInt
