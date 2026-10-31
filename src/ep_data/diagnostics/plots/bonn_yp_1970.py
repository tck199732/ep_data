"""Diagnostic plots for bonn_yp_1970."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl  # noqa: F401
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401

from ep_data.diagnostics.loading import load_dataset
from ep_data.diagnostics.style import diagnostic_style

STYLE_OVERRIDES: dict[str, Any] = {}


@diagnostic_style(STYLE_OVERRIDES)
def plot(input_dir: Path, output_dir: Path) -> None:
    """Generate diagnostic plots for bonn_yp_1970."""
    measurements = load_dataset(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        f"Implement diagnostic plotting for {len(measurements)} measurement(s)"
    )
