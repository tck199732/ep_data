"""Load processed JSON measurements into the shared data model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ep_data.models import Kinematics, Measurement, Metadata, Qualifier, Xsect


def _require_object(value: Any, *, name: str, path: Path) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected {name} to be a JSON object in {path}")
    return value


def load_measurement(path: Path) -> Measurement:
    """Load and validate one processed measurement JSON file."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            raw = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not load processed measurement {path}: {exc}") from exc

    document = _require_object(raw, name="measurement", path=path)
    try:
        xsect_data = _require_object(document["xsect"], name="xsect", path=path)
        kinematics_data = _require_object(
            document["kinematics"], name="kinematics", path=path
        )
        metadata_data = _require_object(
            document["metadata"], name="metadata", path=path
        ).copy()
        raw_qualifiers = metadata_data.pop("qualifiers", [])
        if not isinstance(raw_qualifiers, list):
            raise ValueError(f"Expected qualifiers to be a JSON array in {path}")
        qualifiers = tuple(
            Qualifier(**_require_object(item, name="qualifier", path=path))
            for item in raw_qualifiers
        )
        metadata = Metadata(**metadata_data, qualifiers=qualifiers)
        return Measurement(
            xsect=Xsect(**xsect_data),
            kinematics=Kinematics(**kinematics_data),
            metadata=metadata,
        )
    except KeyError as exc:
        raise ValueError(f"Missing required field {exc.args[0]!r} in {path}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid processed measurement {path}: {exc}") from exc


def load_dataset(input_dir: Path) -> dict[str, Measurement]:
    """Load top-level measurement JSON files, keyed by filename stem."""
    input_dir = Path(input_dir)
    paths = sorted(input_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No processed JSON measurements found in {input_dir}")
    return {path.stem: load_measurement(path) for path in paths}
