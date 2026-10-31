"""Selected exact reproductions of ROOT's predefined colormaps."""

from ep_data.diagnostics.colormaps.factory import available_colormaps, get_cmap
from ep_data.diagnostics.colormaps.registry import (
    COLORMAP_SPECS,
    ROOT_COMMIT,
    ROOT_SOURCE,
    ColormapSpec,
)

kBird = get_cmap("kBird")
kRainBow = get_cmap("kRainBow")
kCherry = get_cmap("kCherry")
kViridis = get_cmap("kViridis")
DEFAULT_CMAP = kBird

__all__ = [
    "COLORMAP_SPECS",
    "DEFAULT_CMAP",
    "ROOT_COMMIT",
    "ROOT_SOURCE",
    "ColormapSpec",
    "available_colormaps",
    "get_cmap",
    "kBird",
    "kCherry",
    "kRainBow",
    "kViridis",
]
