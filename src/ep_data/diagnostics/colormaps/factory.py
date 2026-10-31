"""Load exact ROOT color tables as independent Matplotlib colormaps."""

from __future__ import annotations

from functools import cache
from importlib import resources
from typing import TypeAlias

from matplotlib.colors import ListedColormap

from ep_data.diagnostics.colormaps.registry import COLORMAP_SPECS

RGB: TypeAlias = tuple[float, float, float]
Color: TypeAlias = str | tuple[float, float, float] | tuple[float, float, float, float]


@cache
def _load_colors(name: str) -> tuple[RGB, ...]:
    try:
        spec = COLORMAP_SPECS[name]
    except KeyError as exc:
        available = ", ".join(COLORMAP_SPECS)
        raise KeyError(
            f"Unknown ROOT colormap {name!r}; available: {available}"
        ) from exc

    resource = (
        resources.files("ep_data.diagnostics.colormaps")
        .joinpath("data")
        .joinpath(spec.resource)
    )
    colors: list[RGB] = []
    with resource.open(encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 3:
                raise ValueError(
                    f"Invalid RGB entry in {spec.resource}:{line_number}: {line!r}"
                )
            color = tuple(float(field) for field in fields)
            if any(component < 0.0 or component > 1.0 for component in color):
                raise ValueError(
                    f"RGB component outside [0, 1] in " f"{spec.resource}:{line_number}"
                )
            colors.append(color)

    if len(colors) != 255:
        raise ValueError(
            f"ROOT colormap {name} must contain 255 colors; found {len(colors)}"
        )
    return tuple(colors)


def available_colormaps() -> tuple[str, ...]:
    """Return selected ROOT palette names in registry order."""
    return tuple(COLORMAP_SPECS)


def get_cmap(
    name: str,
    *,
    reversed: bool = False,
    bad: Color = "white",
    under: Color = "white",
    over: Color | None = None,
) -> ListedColormap:
    """Create an independent exact 255-color ROOT palette."""
    colors = _load_colors(name)
    cmap_name = name
    if reversed:
        colors = colors[::-1]
        cmap_name = f"{name}_r"

    cmap = ListedColormap(colors, name=cmap_name)
    if over is None:
        return cmap.with_extremes(bad=bad, under=under)
    return cmap.with_extremes(bad=bad, under=under, over=over)


__all__ = ["available_colormaps", "get_cmap"]
