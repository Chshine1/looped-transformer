from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DataOptions:
    train_file_path: str
    eval_file_path: str
    output_dir: str

    batch_size: int
    delimiter: str
    premise_column: str
    hypothesis_column: str
    label_column: str
    skip_invalid_rows: bool
