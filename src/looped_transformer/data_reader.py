import json
import random
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import override

import pandas as pd
from pandas import DataFrame

from looped_transformer.abstractions.data import MNLIBatch
from looped_transformer.abstractions.training import IDataProvider
from looped_transformer.config.base import (
    ConfigPath,
    NonEmptyString,
    NonNegativeInt,
    OptionsModel,
    PositiveInt,
)


class MNLIDataOptions(OptionsModel):
    seed: NonNegativeInt
    file_path: ConfigPath
    batch_size: PositiveInt
    premise_column: NonEmptyString
    hypothesis_column: NonEmptyString
    label_column: NonEmptyString
    shuffle_cache_dir: ConfigPath
    shuffle_chunk_size: PositiveInt


class MNLIDataProvider(IDataProvider[MNLIBatch]):
    CACHE_FORMAT = "mnli-shuffle-chunks-v1"

    def __init__(self, data_options: MNLIDataOptions):
        self._data_options = data_options
        self._path = data_options.file_path

        self._examples_count: int | None = None
        self._iterator: Iterator[DataFrame] | None = None

    @override
    def reset_source(self, shuffle: bool, seed: int | None) -> None:
        self.close()

        columns = [
            self._data_options.premise_column,
            self._data_options.hypothesis_column,
            self._data_options.label_column,
        ]

        if not shuffle:
            # noinspection argument-list
            self._iterator = pd.read_csv(
                self._path,
                sep="\t",
                dtype=str,
                usecols=columns,
                chunksize=self._data_options.batch_size,
            )
            return

        if seed is None:
            raise ValueError("A seed must be provided")

        chunk_paths = self._prepare_shuffle_chunks()
        rng = random.Random(seed)
        rng.shuffle(chunk_paths)

        def create_iterator():
            pending = pd.DataFrame()
            for chunk_path in chunk_paths:
                # noinspection argument-list
                chunk = pd.read_csv(chunk_path, sep="\t", dtype=str)
                chunk = chunk.sample(
                    frac=1,
                    random_state=rng.randrange(2**32),
                    ignore_index=True,
                )
                if not pending.empty:
                    chunk = pd.concat((pending, chunk), ignore_index=True)

                batch_size = self._data_options.batch_size
                complete_rows = len(chunk) - len(chunk) % batch_size
                for start in range(0, complete_rows, batch_size):
                    yield chunk.iloc[start : start + batch_size]
                pending = chunk.iloc[complete_rows:].copy()

            if not pending.empty:
                yield pending

        self._iterator = create_iterator()

    @override
    def fetch(self) -> MNLIBatch | None:
        if self._iterator is None:
            raise ValueError("A data source must be set first before fetching batches")

        try:
            dataframe = next(self._iterator, None)
        except BaseException:
            self.close()
            raise
        if dataframe is None:
            self.close()
            return None
        return MNLIBatch(
            premises=tuple(dataframe[self._data_options.premise_column].tolist()),
            hypotheses=tuple(dataframe[self._data_options.hypothesis_column].tolist()),
            labels=tuple(dataframe[self._data_options.label_column].tolist()),
        )

    def _get_batches_iterator(
        self,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> Iterator[DataFrame]:
        columns = [
            self._data_options.premise_column,
            self._data_options.hypothesis_column,
            self._data_options.label_column,
        ]

        if not shuffle:
            # noinspection argument-list
            reader = pd.read_csv(
                self._path,
                sep="\t",
                dtype=str,
                usecols=columns,
                chunksize=self._data_options.batch_size,
            )
            with reader:
                yield from reader
            return

        assert seed is not None
        chunk_paths = self._prepare_shuffle_chunks()
        rng = random.Random(seed)
        rng.shuffle(chunk_paths)

        pending = pd.DataFrame()
        for chunk_path in chunk_paths:
            # noinspection argument-list
            chunk = pd.read_csv(chunk_path, sep="\t", dtype=str)
            chunk = chunk.sample(
                frac=1,
                random_state=rng.randrange(2**32),
                ignore_index=True,
            )
            if not pending.empty:
                chunk = pd.concat((pending, chunk), ignore_index=True)

            batch_size = self._data_options.batch_size
            complete_rows = len(chunk) - len(chunk) % batch_size
            for start in range(0, complete_rows, batch_size):
                yield chunk.iloc[start : start + batch_size]
            pending = chunk.iloc[complete_rows:].copy()

        if not pending.empty:
            yield pending

    @override
    def get_samples_count(self) -> int:
        if self._examples_count is not None:
            return self._examples_count

        label_col = self._data_options.label_column

        # noinspection argument-list
        chunks = pd.read_csv(
            self._path,
            sep="\t",
            dtype=str,
            usecols=[label_col],
            chunksize=100_000,
        )
        with chunks:
            count = sum(len(chunk) for chunk in chunks)
        self._examples_count = count
        return count

    def _shuffle_spec(self) -> dict:
        from looped_transformer.artifacts import get_file_fingerprint

        return {
            "format": self.CACHE_FORMAT,
            "source": get_file_fingerprint(self._path),
            "columns": [
                self._data_options.premise_column,
                self._data_options.hypothesis_column,
                self._data_options.label_column,
            ],
            "chunk_size": self._data_options.shuffle_chunk_size,
        }

    def _prepare_shuffle_chunks(self) -> list[Path]:
        from looped_transformer.artifacts import get_artifact_id

        spec = self._shuffle_spec()
        identifier = get_artifact_id(spec)
        cache_dir = Path(self._data_options.shuffle_cache_dir) / identifier
        manifest_path = cache_dir / "manifest.json"

        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                names = manifest["chunks"]
                paths = [cache_dir / name for name in names]
                if (
                    manifest.get("id") == identifier
                    and manifest.get("spec") == spec
                    and all(path.is_file() for path in paths)
                ):
                    return paths
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_dir.parent / f".{identifier}.{uuid.uuid4().hex}.tmp"
        temporary.mkdir()
        chunk_names: list[str] = []
        try:
            # noinspection argument-list
            reader = pd.read_csv(
                self._path,
                sep="\t",
                dtype=str,
                usecols=spec["columns"],
                chunksize=self._data_options.shuffle_chunk_size,
            )
            with reader:
                for index, chunk in enumerate(reader):
                    name = f"chunk-{index:06d}.tsv"
                    chunk.to_csv(
                        temporary / name,
                        sep="\t",
                        index=False,
                        encoding="utf-8",
                        lineterminator="\n",
                    )
                    chunk_names.append(name)

            manifest = {"id": identifier, "spec": spec, "chunks": chunk_names}
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(cache_dir)
        except FileExistsError:
            shutil.rmtree(temporary, ignore_errors=True)
            return self._prepare_shuffle_chunks()
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return [cache_dir / name for name in chunk_names]

    @override
    def get_batch_size(self) -> int:
        return self._data_options.batch_size

    @override
    def close(self) -> None:
        iterator, self._iterator = self._iterator, None
        if iterator is None:
            return
        close = getattr(iterator, "close", None)
        if close is not None:
            close()
