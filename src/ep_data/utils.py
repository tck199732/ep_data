"""Small framework-owned filesystem and fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


def fingerprint_tree(root: Path, *, excluded_names: set[str] | None = None) -> str:
    """Hash relative file names and contents in deterministic order."""
    excluded_names = excluded_names or set()
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in excluded_names
    )
    for path in files:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def fingerprint_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(jsonify(value), stream, indent=2, sort_keys=True)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def jsonify(obj: Any) -> Any:
    """Recursively convert supported objects into JSON-compatible values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {item.name: jsonify(getattr(obj, item.name)) for item in fields(obj)}
    if isinstance(obj, np.ndarray):
        if obj.ndim != 1:
            raise ValueError("Only one-dimensional NumPy arrays are supported")
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(key): jsonify(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonify(item) for item in obj]
    return obj


def atomic_replace_directory(source: Path, target: Path) -> None:
    """Replace a directory while restoring the previous target on commit failure."""
    import shutil
    import uuid

    source = Path(source)
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    backup = target.with_name(f".{target.name}.backup-{uuid.uuid4().hex}")
    had_target = target.exists()
    if had_target:
        os.replace(target, backup)
    try:
        os.replace(source, target)
    except Exception:
        if had_target and backup.exists():
            os.replace(backup, target)
        raise
    else:
        if had_target:
            shutil.rmtree(backup)
