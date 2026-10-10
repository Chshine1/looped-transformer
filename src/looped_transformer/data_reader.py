from __future__ import annotations

import json
import random
import shutil
import uuid
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

from looped_transformer.artifacts import get_artifact_id, get_file_fingerprint
from looped_transformer.config import DataOptions, RuntimeOptions
from looped_transformer.types.data import DataSplit, MNLIBatch


class MNLIDataReader:
    CACHE_FORMAT = "mnli-shuffle-chunks-v1"

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
        seed: int | None = None,
    ) -> Iterator[MNLIBatch]:
        path = self._paths[split]
        columns = [
            self._data_options.premise_column,
            self._data_options.hypothesis_column,
            self._data_options.label_column,
        ]

        if not shuffle:
            # noinspection argument-list
            reader = pd.read_csv(
                path,
                sep="\t",
                dtype=str,
                usecols=columns,
                chunksize=self._data_options.batch_size,
            )
            for chunk in reader:
                yield self._to_batch(chunk)
            return

        chunk_paths = self._prepare_shuffle_chunks(split)
        rng = random.Random(self._runtime_options.seed if seed is None else seed)
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
                yield self._to_batch(chunk.iloc[start : start + batch_size])
            pending = chunk.iloc[complete_rows:].copy()

        if not pending.empty:
            yield self._to_batch(pending)

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

    def _shuffle_spec(self, split: DataSplit) -> dict:
        path = self._paths[split]
        return {
            "format": self.CACHE_FORMAT,
            "source": get_file_fingerprint(path),
            "columns": [
                self._data_options.premise_column,
                self._data_options.hypothesis_column,
                self._data_options.label_column,
            ],
            "chunk_size": self._data_options.shuffle_chunk_size,
        }

    def _prepare_shuffle_chunks(self, split: DataSplit) -> list[Path]:
        spec = self._shuffle_spec(split)
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
                self._paths[split],
                sep="\t",
                dtype=str,
                usecols=spec["columns"],
                chunksize=self._data_options.shuffle_chunk_size,
            )
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
            # Another process completed the identical cache first.
            shutil.rmtree(temporary, ignore_errors=True)
            return self._prepare_shuffle_chunks(split)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

        return [cache_dir / name for name in chunk_names]
