"""Shared Matplotlib defaults for diagnostic plots."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any, ParamSpec, TypeVar

import matplotlib as mpl
from cycler import cycler

P = ParamSpec("P")
R = TypeVar("R")

DEFAULT_CMAP_PETROFF = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
    "#21dd7a",
]

DEFAULT_MPL_CONFIG: dict[str, Any] = {
    "axes.prop_cycle": cycler("color", DEFAULT_CMAP_PETROFF),
    # Fonts
    "font.family": "sans-serif",
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial"],
    "font.size": 14,
    "mathtext.fontset": "stixsans",
    "mathtext.default": "rm",
    # Figure layout
    "figure.dpi": 200,
    "figure.figsize": (4, 3.5),
    "figure.facecolor": "white",
    "figure.subplot.bottom": 0.16,
    "figure.subplot.top": 0.93,
    "figure.subplot.left": 0.16,
    "figure.subplot.right": 0.95,
    # Axes
    "axes.linewidth": 1.0,
    "axes.labelsize": 32,
    "axes.labelpad": 24,
    "xaxis.labellocation": "right",
    "yaxis.labellocation": "top",
    # Lines and markers
    "lines.linewidth": 1.0,
    "lines.markersize": 8,
    # X ticks
    "xtick.top": True,
    "xtick.labelsize": 25,
    "xtick.major.size": 10,
    "xtick.minor.size": 5,
    "xtick.direction": "in",
    "xtick.minor.visible": True,
    # Y ticks
    "ytick.right": True,
    "ytick.labelsize": 25,
    "ytick.major.size": 14,
    "ytick.minor.size": 7,
    "ytick.direction": "in",
    "ytick.minor.visible": True,
    # Legend
    "legend.loc": "best",
    "legend.numpoints": 1,
    "legend.fontsize": 28,
    "legend.labelspacing": 0.3,
    "legend.frameon": False,
    # Export
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
}


def diagnostic_style(
    overrides: Mapping[str, Any] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a plotting function with scoped Matplotlib configuration."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            config = DEFAULT_MPL_CONFIG | dict(overrides or {})
            with mpl.rc_context(config):
                return function(*args, **kwargs)

        return wrapped

    return decorator
