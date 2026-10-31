"""Non-destructive scaffolding for dataset diagnostics modules."""

from __future__ import annotations

import keyword
import os
import tempfile
from pathlib import Path

PLOTS_DIR = Path(__file__).resolve().parent / "plots"

_TEMPLATE = '''\
"""Diagnostic plots for {key}."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401

from ep_data.diagnostics.loading import load_dataset
from ep_data.diagnostics.style import diagnostic_style

STYLE_OVERRIDES: dict[str, Any] = {{}}


@diagnostic_style(STYLE_OVERRIDES)
def plot(input_dir: Path, output_dir: Path) -> None:
    """Generate diagnostic plots for {key}."""
    measurements = load_dataset(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        f"Implement diagnostic plotting for {{len(measurements)}} measurement(s)"
    )
'''


def create_diagnostics_template(
    key: str,
    destination_dir: Path | None = None,
) -> Path | None:
    """Create a diagnostics module once, returning None when it already exists."""
    if not key.isidentifier() or keyword.iskeyword(key):
        raise ValueError(f"Dataset key is not a valid Python module name: {key!r}")

    directory = Path(destination_dir) if destination_dir is not None else PLOTS_DIR
    destination = directory / f"{key}.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{key}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(_TEMPLATE.format(key=key))
        try:
            os.link(temporary, destination)
        except FileExistsError:
            return None
        return destination
    finally:
        temporary.unlink(missing_ok=True)
