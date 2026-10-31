"""Diagnostic plots for slac_yp_1969."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from ep_data.diagnostics.loading import load_dataset
from ep_data.diagnostics.style import DEFAULT_CMAP_PETROFF, diagnostic_style
from ep_data.models import Measurement
from ep_data.parser.physics import MASS_LAMBDA, get_dOmega_dt

STYLE_OVERRIDES: dict[str, Any] = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.labelpad": 8,
    "axes.titlesize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
_SERIES = (
    ("lambda_5gev", 5, "o", DEFAULT_CMAP_PETROFF[0]),
    ("lambda_8gev", 8, "s", DEFAULT_CMAP_PETROFF[1]),
    ("lambda_11gev", 11, "^", DEFAULT_CMAP_PETROFF[2]),
    ("lambda_16gev", 16, "D", DEFAULT_CMAP_PETROFF[3]),
)
_X_LABEL = r"$\mathrm{-t}\ \mathrm{[GeV^2]}$"
_Y_LABEL = r"$\mathrm{d\sigma/dt}\ \mathrm{[\mu b/GeV^2]}$"
_TITLE = r"$\mathrm{\gamma p\rightarrow K^{+}\Lambda}$"
_NB_TO_MICROBARN = 1.0e-3


@diagnostic_style(STYLE_OVERRIDES)
def plot(input_dir: Path, output_dir: Path) -> None:
    """Generate diagnostic plots for slac_yp_1969."""
    measurements = load_dataset(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    missing = [name for name, *_ in _SERIES if name not in measurements]
    if missing:
        raise ValueError(f"Processed data is missing {', '.join(missing)}")
    _plot_sigmaU_t(
        measurements,
        output_dir / "lambda_sigmaU_t.png",
    )


def _combined_uncertainty(measurement: Measurement) -> np.ndarray:
    """Return lower and upper uncertainties combined in quadrature."""
    xsect = measurement.xsect
    if xsect.sigU is None or xsect.sigU_err is None:
        raise ValueError("SLAC Lambda measurement has no sigU data")
    if xsect.sigU_sys_err_low is None or xsect.sigU_sys_err_high is None:
        raise ValueError("SLAC Lambda measurement has no sigU systematic errors")

    statistical = np.asarray(xsect.sigU_err)
    systematic_low = np.asarray(xsect.sigU_sys_err_low)
    systematic_high = np.asarray(xsect.sigU_sys_err_high)
    return np.vstack(
        (
            np.hypot(statistical, systematic_low),
            np.hypot(statistical, systematic_high),
        )
    )


def _plot_sigmaU_t(
    measurements: dict[str, Measurement],
    output_path: Path,
) -> None:
    """Plot Lambda sigU versus minus-t for the four photon energies."""
    fig, axis = plt.subplots(figsize=(6, 4.8), constrained_layout=True)
    try:
        for name, energy, marker, color in _SERIES:
            measurement = measurements[name]
            if measurement.kinematics.t is None or measurement.xsect.sigU is None:
                raise ValueError(f"{name} has no t or sigU data")

            minus_t = np.asarray(measurement.kinematics.t)
            sig_u, uncertainty = _to_dsigma_dt(measurement)
            if minus_t.shape != sig_u.shape or uncertainty.shape != (2, sig_u.size):
                raise ValueError(f"{name} arrays do not have matching shapes")
            if not np.all(np.isfinite(minus_t)) or not np.all(np.isfinite(sig_u)):
                raise ValueError(f"{name} contains non-finite data")
            if not np.all(np.isfinite(uncertainty)):
                raise ValueError(f"{name} contains non-finite uncertainties")
            if np.any(sig_u <= 0.0):
                raise ValueError(f"{name} contains non-positive sigU values")

            order = np.argsort(minus_t)
            axis.errorbar(
                minus_t[order],
                sig_u[order],
                yerr=uncertainty[:, order],
                fmt=marker,
                color=color,
                ecolor=color,
                markerfacecolor="none",
                markersize=5.0,
                capsize=2.0,
                elinewidth=0.9,
                linestyle="none",
                label=rf"$\mathrm{{E}}_\mathrm{{\gamma}}={energy}\ \mathrm{{GeV}}$",
            )

        axis.set_yscale("log")
        axis.set_xlim(0.0, 2.1)
        axis.set_ylim(1e-4, 1e0)
        axis.set_xlabel(_X_LABEL)
        axis.set_ylabel(_Y_LABEL)
        axis.set_title(_TITLE)
        axis.legend()

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def _to_dsigma_dt(measurement: Measurement) -> tuple[np.ndarray, np.ndarray]:
    """Convert parsed nb/sr values and uncertainties to microbarn/GeV^2."""
    if measurement.xsect.sigU is None:
        raise ValueError("SLAC Lambda measurement has no sigU data")

    q2 = np.asarray(measurement.kinematics.Q2)
    w = np.asarray(measurement.kinematics.W)
    sig_u = np.asarray(measurement.xsect.sigU)
    uncertainty = _combined_uncertainty(measurement)
    if q2.shape != sig_u.shape or w.shape != sig_u.shape:
        raise ValueError("SLAC Lambda kinematic and cross-section shapes do not match")

    conversion = get_dOmega_dt(q2, w, MASS_LAMBDA) * _NB_TO_MICROBARN
    return sig_u * conversion, uncertainty * conversion[np.newaxis, :]
