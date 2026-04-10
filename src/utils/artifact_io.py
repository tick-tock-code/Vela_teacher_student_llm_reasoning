from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(payload: Any) -> Any:
    if is_dataclass(payload):
        return asdict(payload)
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, dict):
        return {str(key): to_jsonable(value) for key, value in payload.items()}
    if isinstance(payload, list):
        return [to_jsonable(value) for value in payload]
    return payload


def write_json(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    return path


def write_text(path: Path, content: str) -> Path:
    ensure_dir(path.parent)
    path.write_text(content, encoding="utf-8")
    return path


def write_markdown(path: Path, content: str) -> Path:
    if not content.endswith("\n"):
        content += "\n"
    return write_text(path, content)


def write_csv(path: Path, frame: pd.DataFrame) -> Path:
    ensure_dir(path.parent)
    frame.to_csv(path, index=False)
    return path


def timestamped_run_dir(root: Path, label: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return ensure_dir(root / f"{stamp}_{label}")
