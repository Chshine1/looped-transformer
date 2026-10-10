from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataOptions:
    train_file_path: str
    eval_file_path: str
    output_dir: str

    batch_size: int
    premise_column: str
    hypothesis_column: str
    label_column: str

    shuffle_cache_dir: str
    shuffle_chunk_size: int
