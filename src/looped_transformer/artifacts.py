from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any


def _canonicalize(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _canonicalize(asdict(value))
    if isinstance(value, dict):
        return {str(key): _canonicalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted((_canonicalize(item) for item in value), key=repr)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    return value


def _get_canonical_json(value: Any) -> str:
    return json.dumps(
        _canonicalize(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def get_artifact_id(value: Any, length: int = 16) -> str:
    digest = hashlib.sha256(_get_canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


@lru_cache(maxsize=16)
def _sha256(path: str, size: int, modified_ns: int) -> str:
    del size, modified_ns  # They form part of the cache key.
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def get_file_fingerprint(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    return {
        "size": stat.st_size,
        "sha256": _sha256(str(resolved), stat.st_size, stat.st_mtime_ns),
    }


def _get_run_artifact_spec(runtime: Any, data: Any, tokenizer: Any, model: Any) -> dict:
    data_config = asdict(data)
    data_config.pop("output_dir", None)
    data_config.pop("shuffle_cache_dir", None)
    data_config.pop("train_file_path", None)
    data_config.pop("eval_file_path", None)
    data_config["train_source"] = get_file_fingerprint(data.train_file_path)
    data_config["eval_source"] = get_file_fingerprint(data.eval_file_path)
    return {
        "format": "looped-transformer-training-v1",
        "runtime": runtime,
        "data": data_config,
        "tokenizer": tokenizer,
        "model": model,
    }


def get_run_output_dir(
    runtime: Any, data: Any, tokenizer: Any, model: Any
) -> tuple[Path, str, dict]:
    spec = _get_run_artifact_spec(runtime, data, tokenizer, model)
    identifier = get_artifact_id(spec)
    return Path(data.output_dir) / identifier, identifier, spec


def write_manifest(directory: Path, identifier: str, spec: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "artifact.json"
    contents = json.dumps(
        {"id": identifier, "spec": _canonicalize(spec)},
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != json.loads(contents):
            raise ValueError(
                f"Artifact manifest does not match directory: {directory}"
            ) from None
        return
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(contents + "\n", encoding="utf-8")
    try:
        temporary.replace(path)
    except FileExistsError:
        temporary.unlink(missing_ok=True)
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != json.loads(contents):
            raise ValueError(
                f"Artifact manifest does not match directory: {directory}"
            ) from None
