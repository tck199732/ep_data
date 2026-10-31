import hashlib
from importlib import resources

import numpy as np
import pytest
from matplotlib.colors import ListedColormap

from ep_data.diagnostics import cmap
from ep_data.diagnostics.colormaps import (
    COLORMAP_SPECS,
    DEFAULT_CMAP,
    ROOT_COMMIT,
    available_colormaps,
    get_cmap,
    kBird,
    kCherry,
    kRainBow,
    kViridis,
)

EXPECTED_HASHES = {
    "k_bird.txt": "4051e5a4f497463fbcbfd50be7d627a2a6e271c21471483f329873c6108afeb3",
    "k_cherry.txt": "ac1e76018c00d6cfa601dcd918c7a66dc12dff271500e8e03ff526d8050fccd9",
    "k_rainbow.txt": "5264255f17f32f92f48d5e8e0e65d929d7d0877f3652c135dbc9d15821d834be",
    "k_viridis.txt": "876775dde57685d1499a7cec31448e48d15f8cdf70991c2c1c9c2643bf88119e",
}


def test_selected_colormaps_use_canonical_root_names_and_ids():
    assert available_colormaps() == ("kBird", "kRainBow", "kCherry", "kViridis")
    assert {name: spec.root_id for name, spec in COLORMAP_SPECS.items()} == {
        "kBird": 57,
        "kRainBow": 55,
        "kCherry": 75,
        "kViridis": 112,
    }
    assert not hasattr(cmap, "kRainbow")


@pytest.mark.parametrize("name", available_colormaps())
def test_colormaps_are_exact_255_color_tables(name):
    result = get_cmap(name)

    assert isinstance(result, ListedColormap)
    assert result.name == name
    assert result.N == 255
    assert len(result.colors) == 255
    np.testing.assert_allclose(result.get_bad(), [1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(result.get_under(), [1.0, 1.0, 1.0, 1.0])


def test_exported_colormaps_and_default():
    assert [item.name for item in (kBird, kRainBow, kCherry, kViridis)] == [
        "kBird",
        "kRainBow",
        "kCherry",
        "kViridis",
    ]
    assert DEFAULT_CMAP is kBird
    assert cmap.DEFAULT_CMAP is kBird


def test_get_cmap_returns_independent_mutable_instances():
    first = get_cmap("kBird")
    second = get_cmap("kBird")

    assert first is not second
    first = first.with_extremes(bad="black")
    np.testing.assert_allclose(first.get_bad(), [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(second.get_bad(), [1.0, 1.0, 1.0, 1.0])


def test_reversed_palette_preserves_exact_color_order():
    forward = get_cmap("kCherry")
    reversed_cmap = get_cmap("kCherry", reversed=True)

    assert reversed_cmap.name == "kCherry_r"
    np.testing.assert_array_equal(reversed_cmap.colors, forward.colors[::-1])


def test_unknown_colormap_lists_canonical_names():
    with pytest.raises(KeyError, match="Unknown ROOT colormap.*kRainBow"):
        get_cmap("kRainbow")


def test_palette_resources_match_pinned_checksums():
    data = resources.files("ep_data.diagnostics.colormaps").joinpath("data")
    actual = {
        filename: hashlib.sha256(data.joinpath(filename).read_bytes()).hexdigest()
        for filename in EXPECTED_HASHES
    }

    assert actual == EXPECTED_HASHES
    assert ROOT_COMMIT == "6f7cb11d8273fbda7be4cffa3b968d9a19f6e9f7"
