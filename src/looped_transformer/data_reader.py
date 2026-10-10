from typing import Iterator

import pandas as pd

from looped_transformer.config import DataOptions, RuntimeOptions
from looped_transformer.types.data import DataSplit, MNLIBatch


class MNLIDataReader:
    def __init__(self, runtime_options: RuntimeOptions, data_options: DataOptions):
        self._runtime_options = runtime_options
        self._data_options = data_options
        self._paths = {
            DataSplit.TRAIN: data_options.train_file_path,
            DataSplit.EVAL: data_options.eval_file_path,
        }

    def get_batches_iterator(
        self,
        split: DataSplit,
        shuffle: bool = True,
    ) -> Iterator[MNLIBatch]:
        path = self._paths[split]

        if not shuffle:
            # noinspection argument-list
            reader = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                usecols=[
                    self._data_options.premise_column,
                    self._data_options.hypothesis_column,
                    self._data_options.label_column,
                ],
                chunksize=self._data_options.batch_size,
            )
            for chunk in reader:
                yield self._to_batch(chunk)
            return

        # noinspection argument-list
        raise NotImplementedError

    def count_examples(self, split: DataSplit) -> int:
        path = self._paths[split]
        label_col = self._data_options.label_column

        # noinspection argument-list
        chunks = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            usecols=[label_col],
            chunksize=100_000,
        )
        return sum(len(chunk) for chunk in chunks)

    def _to_batch(self, df) -> MNLIBatch:
        return MNLIBatch(
            premises=tuple(df[self._data_options.premise_column].tolist()),
            hypotheses=tuple(df[self._data_options.hypothesis_column].tolist()),
            labels=tuple(df[self._data_options.label_column].tolist()),
        )
