from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from src.utils.artifact_io import ensure_dir, read_json, write_json
from src.utils.paths import SAVED_MODEL_CONFIGS_DIR


BUNDLE_MANIFEST_FILENAME = "bundle_manifest.json"


def bundle_dir_from_run_id(run_id: str) -> Path:
    return SAVED_MODEL_CONFIGS_DIR / str(run_id).strip()


def list_saved_bundle_dirs() -> list[Path]:
    if not SAVED_MODEL_CONFIGS_DIR.exists():
        return []
    return sorted(
        [path for path in SAVED_MODEL_CONFIGS_DIR.iterdir() if path.is_dir()],
        key=lambda item: item.name,
        reverse=True,
    )


def resolve_saved_bundle_path(path_or_id: str | Path) -> Path:
    candidate = Path(path_or_id)
    if candidate.exists():
        return candidate
    fallback = bundle_dir_from_run_id(str(path_or_id))
    if fallback.exists():
        return fallback
    return candidate


def write_bundle_manifest(bundle_dir: Path, manifest: dict[str, Any]) -> Path:
    ensure_dir(bundle_dir)
    return write_json(bundle_dir / BUNDLE_MANIFEST_FILENAME, manifest)


def load_bundle_manifest(bundle_dir_or_id: str | Path) -> tuple[Path, dict[str, Any]]:
    bundle_dir = resolve_saved_bundle_path(bundle_dir_or_id)
    manifest_path = bundle_dir / BUNDLE_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise RuntimeError(f"Saved bundle manifest not found: {manifest_path}")
    return bundle_dir, read_json(manifest_path)


def save_pickle(path: Path, payload: Any) -> Path:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)
