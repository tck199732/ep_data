"""Metadata for the selected ROOT colormaps."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

ROOT_COMMIT: Final = "6f7cb11d8273fbda7be4cffa3b968d9a19f6e9f7"
ROOT_SOURCE: Final = (
    "https://github.com/root-project/root/blob/"
    f"{ROOT_COMMIT}/core/base/src/TColor.cxx"
)


@dataclass(frozen=True, slots=True)
class ColormapSpec:
    """Provenance and resource information for one ROOT palette."""

    name: str
    root_id: int
    resource: str
    cvd_friendly: bool


_SPECS = {
    "kBird": ColormapSpec("kBird", 57, "k_bird.txt", True),
    "kRainBow": ColormapSpec("kRainBow", 55, "k_rainbow.txt", False),
    "kCherry": ColormapSpec("kCherry", 75, "k_cherry.txt", True),
    "kViridis": ColormapSpec("kViridis", 112, "k_viridis.txt", True),
}

COLORMAP_SPECS = MappingProxyType(_SPECS)

__all__ = ["COLORMAP_SPECS", "ROOT_COMMIT", "ROOT_SOURCE", "ColormapSpec"]
