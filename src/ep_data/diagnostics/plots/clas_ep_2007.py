"""Diagnostic plots for clas_ep_2007."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np

from ep_data.diagnostics.loading import load_dataset
from ep_data.diagnostics.style import diagnostic_style
from ep_data.models import Measurement

STYLE_OVERRIDES: dict[str, Any] = {
    "font.sans-serif": ["TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.labelpad": 8,
    "axes.titlesize": 12,
    "figure.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
_Q2 = 0.65
_W_VALUES = (1.65, 1.725, 1.775, 1.825, 1.875, 1.925, 1.975, 2.05)
_RESPONSES = (
    ("sigU", r"$\mathrm{\sigma}_{\mathrm{U}}\ \mathrm{[nb/sr]}$", False),
    ("sigLT", r"$\mathrm{\sigma}_{\mathrm{LT}}\ \mathrm{[nb/sr]}$", True),
    ("sigTT", r"$\mathrm{\sigma}_{\mathrm{TT}}\ \mathrm{[nb/sr]}$", True),
)
_X_LABEL = r"$\mathrm{\cos}(\mathrm{\theta}_{K}^{*})$"
_Q2_LABEL = r"$\mathrm{Q}^{2}=0.65\ \mathrm{GeV}^{2}$"
_SIGMA_U_Q2_VALUES = (1.0, 1.55, 2.05, 2.55)
_SIGMA_U_COSINE_VALUES = (-0.25, 0.35)
_W_LABEL = r"$\mathrm{W}\ \mathrm{[GeV]}$"
_SIGMA_U_LABEL = r"$\mathrm{\sigma}_{\mathrm{U}}\ \mathrm{[nb/sr]}$"


@diagnostic_style(STYLE_OVERRIDES)
def plot(input_dir: Path, output_dir: Path) -> None:
    """Generate diagnostic plots for clas_ep_2007."""
    measurements = load_dataset(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        fine_w = measurements["lambda_e2p567_fine_w"]
        e4 = measurements["lambda_e4"]
    except KeyError as exc:
        raise ValueError(f"Processed data is missing {exc.args[0]}") from exc
    _plot_sigma_ct(
        fine_w,
        output_dir / "lambda_q2_0p65_sigma_ct.png",
    )
    _plot_sigmaU_W(
        e4,
        output_dir / "lambda_sigmaU_w.png",
    )


def _plot_sigma_ct(
    measurement: Measurement,
    output_path: Path,
) -> None:
    """Plot the three unseparated responses versus the kaon CM-angle cosine."""
    kinematics = measurement.kinematics
    if kinematics.thetacm_center is None:
        raise ValueError("lambda_e2p567_fine_w has no thetacm_center values")

    q2 = np.asarray(kinematics.Q2)
    w_values = np.asarray(kinematics.W)
    theta_centers = np.asarray(kinematics.thetacm_center)
    expected_shape = q2.shape
    if w_values.shape != expected_shape or theta_centers.shape != expected_shape:
        raise ValueError("CLAS kinematic arrays do not have matching shapes")

    fig, axes = plt.subplots(
        3,
        8,
        figsize=(18, 9),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    try:
        for column, w_center in enumerate(_W_VALUES):
            selected = np.flatnonzero(
                np.isclose(q2, _Q2) & np.isclose(w_values, w_center)
            )
            if selected.size != 6:
                raise ValueError(
                    f"Expected 6 points at Q2={_Q2:g}, W={w_center:g}; "
                    f"found {selected.size}"
                )

            cosine = np.cos(theta_centers[selected])
            order = np.argsort(cosine)
            cosine = cosine[order]
            selected = selected[order]
            if not np.all(np.isfinite(cosine)):
                raise ValueError(f"Non-finite angle at W={w_center:g}")

            for row, (component, y_label, show_zero) in enumerate(_RESPONSES):
                axis = axes[row, column]
                values = getattr(measurement.xsect, component)
                errors = getattr(measurement.xsect, f"{component}_err")
                if values is None or errors is None:
                    raise ValueError(f"lambda_e2p567_fine_w has no {component} data")

                response = values[selected]
                uncertainty = errors[selected]
                if not np.all(np.isfinite(response)) or not np.all(
                    np.isfinite(uncertainty)
                ):
                    raise ValueError(f"Non-finite {component} data at W={w_center:g}")

                if show_zero:
                    axis.axhline(0.0, color="0.55", linewidth=0.8, zorder=0)
                axis.errorbar(
                    cosine,
                    response,
                    yerr=uncertainty,
                    fmt="o",
                    color="black",
                    ecolor="black",
                    markersize=3.5,
                    capsize=2.0,
                    elinewidth=0.8,
                    linestyle="none",
                )
                axis.set_xlim(-1.0, 1.0)
                axis.set_xticks((-1.0, -0.5, 0.0, 0.5, 1.0))
                if column == 0:
                    axis.set_ylabel(y_label)
                if row == 2:
                    axis.set_xlabel(_X_LABEL)

            axes[0, column].set_title(rf"$\mathrm{{W}}={w_center:g}\ \mathrm{{GeV}}$")

        fig.suptitle(_Q2_LABEL)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)


def _plot_sigmaU_W(
    measurement: Measurement,
    output_path: Path,
) -> None:
    """Plot the unseparated response versus W at fixed Q2 and CM angle."""
    kinematics = measurement.kinematics
    if kinematics.thetacm_center is None:
        raise ValueError("lambda_e4 has no thetacm_center values")
    if measurement.xsect.sigU is None or measurement.xsect.sigU_err is None:
        raise ValueError("lambda_e4 has no sigU data")

    q2_values = np.asarray(kinematics.Q2)
    w_values = np.asarray(kinematics.W)
    cosine_values = np.cos(np.asarray(kinematics.thetacm_center))
    sig_u = np.asarray(measurement.xsect.sigU)
    sig_u_errors = np.asarray(measurement.xsect.sigU_err)
    expected_shape = q2_values.shape
    arrays = (w_values, cosine_values, sig_u, sig_u_errors)
    if any(array.shape != expected_shape for array in arrays):
        raise ValueError("CLAS lambda_e4 arrays do not have matching shapes")

    fig, axes = plt.subplots(
        4,
        2,
        figsize=(9, 12),
        sharex=True,
        sharey="row",
        constrained_layout=True,
    )
    try:
        for row, q2_center in enumerate(_SIGMA_U_Q2_VALUES):
            for column, cosine_center in enumerate(_SIGMA_U_COSINE_VALUES):
                selected = np.flatnonzero(
                    np.isclose(q2_values, q2_center)
                    & np.isclose(cosine_values, cosine_center)
                )
                expected_points = 6 if np.isclose(q2_center, 2.55) else 8
                if selected.size != expected_points:
                    raise ValueError(
                        f"Expected {expected_points} points at Q2={q2_center:g}, "
                        f"cos(theta)={cosine_center:g}; found {selected.size}"
                    )

                order = np.argsort(w_values[selected])
                selected = selected[order]
                panel_w = w_values[selected]
                response = sig_u[selected]
                uncertainty = sig_u_errors[selected]
                if not np.all(np.isfinite(panel_w)) or not np.all(
                    np.isfinite(response)
                ):
                    raise ValueError(
                        f"Non-finite sigU data at Q2={q2_center:g}, "
                        f"cos(theta)={cosine_center:g}"
                    )
                if not np.all(np.isfinite(uncertainty)):
                    raise ValueError(
                        f"Non-finite sigU uncertainty at Q2={q2_center:g}, "
                        f"cos(theta)={cosine_center:g}"
                    )
                if np.any(np.diff(panel_w) <= 0.0):
                    raise ValueError(
                        f"W values are not unique and increasing at Q2={q2_center:g}, "
                        f"cos(theta)={cosine_center:g}"
                    )

                axis = axes[row, column]
                axis.errorbar(
                    panel_w,
                    response,
                    yerr=uncertainty,
                    fmt="o",
                    color="black",
                    ecolor="black",
                    markersize=3.5,
                    capsize=2.0,
                    elinewidth=0.8,
                    linestyle="none",
                )
                axis.set_xlim(1.6, 2.4)
                axis.set_xticks((1.6, 1.8, 2.0, 2.2, 2.4))
                if column == 0:
                    axis.set_ylabel(_SIGMA_U_LABEL)
                    axis.text(
                        0.04,
                        0.92,
                        rf"$\mathrm{{Q}}^2={q2_center:g}\ \mathrm{{GeV}}^2$",
                        transform=axis.transAxes,
                        va="top",
                    )
                if row == len(_SIGMA_U_Q2_VALUES) - 1:
                    axis.set_xlabel(_W_LABEL)
                if row == 0:
                    axis.set_title(
                        rf"$\mathrm{{\cos}}(\mathrm{{\theta}}_K^*)="
                        rf"{cosine_center:g}$"
                    )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
    finally:
        plt.close(fig)
