import json
import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


class StructuredEvaluationLogger:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.batch_log_path = self.output_dir / "batches.jsonl"

    def start(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_log_path.write_text("", encoding="utf-8")

    def log_batch(self, diagnostics: Any) -> None:
        record = {
            "event": "evaluation_batch",
            "diagnostics": self._to_json_value(diagnostics),
        }
        with self.batch_log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            stream.write("\n")

    @classmethod
    def _to_json_value(cls, value: Any) -> Any:
        if is_dataclass(value) and not isinstance(value, type):
            return cls._to_json_value(asdict(value))
        if isinstance(value, dict):
            return {str(key): cls._to_json_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_json_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, float) and not math.isfinite(value):
            return str(value)
        return value
