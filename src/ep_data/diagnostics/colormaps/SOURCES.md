# ROOT colormap provenance

The color tables in `data/` reproduce selected high-quality palettes from
ROOT's `TColor::SetPalette` implementation.

- Repository: <https://github.com/root-project/root>
- Branch at selection time: `master`
- Commit: `6f7cb11d8273fbda7be4cffa3b968d9a19f6e9f7`
- Source: `core/base/src/TColor.cxx`
- Colors per palette: 255

The selected canonical ROOT names and palette identifiers are:

| Name | ROOT ID | CVD friendly in ROOT documentation |
|---|---:|:---:|
| `kBird` | 57 | yes |
| `kRainBow` | 55 | no |
| `kCherry` | 75 | yes |
| `kViridis` | 112 | yes |

`scripts/generate_root_colormaps.py` contains the pinned gradient anchors and
reproduces ROOT's interval sizing, interpolation, and conversion to `Float_t`.
The generated text files store one RGB float triplet per line and require no
ROOT or PyROOT installation at runtime.
