from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class ArtifactSpec:
    """Domain-neutral description used to identify a reproducible artifact."""

    format: str
    values: Mapping[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {"format": self.format, **self.values}


def _canonicalize(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _canonicalize(value.model_dump())
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
    if isinstance(value, torch.device | torch.dtype):
        return str(value)
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
def _sha256(path: str, _size: int, _modified_ns: int) -> str:
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


def get_run_output_dir(
    output_dir: Path, artifact_spec: ArtifactSpec
) -> tuple[Path, str, dict]:
    spec = artifact_spec.as_dict()
    identifier = get_artifact_id(spec)
    return output_dir / identifier, identifier, spec


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
