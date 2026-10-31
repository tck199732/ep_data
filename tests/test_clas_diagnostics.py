from pathlib import Path

import numpy as np

from ep_data.diagnostics.plots.clas_ep_2007 import (
    _plot_sigma_ct,
    _plot_sigmaU_W,
)
from ep_data.models import Kinematics, Measurement, Metadata, Xsect


def _measurement() -> Measurement:
    w_centers = np.asarray((1.65, 1.725, 1.775, 1.825, 1.875, 1.925, 1.975, 2.05))
    cosine = np.asarray((-0.6, -0.25, 0.05, 0.35, 0.65, 0.9))
    points = w_centers.size * cosine.size
    values = np.linspace(-20.0, 100.0, points)
    errors = np.full(points, 2.0)
    return Measurement(
        xsect=Xsect(
            sigU=values + 30.0,
            sigU_err=errors,
            sigLT=values,
            sigLT_err=errors,
            sigTT=-values,
            sigTT_err=errors,
        ),
        kinematics=Kinematics(
            Q2=np.full(points, 0.65),
            W=np.repeat(w_centers, cosine.size),
            thetacm_center=np.tile(np.arccos(cosine), w_centers.size),
        ),
        metadata=Metadata(
            reaction="E- P --> E- K+ LAMBDA X",
            unit="nb/sr",
            angle_unit="rad",
        ),
    )


def test_plot_sigma_ct_writes_png(tmp_path: Path):
    output = tmp_path / "lambda_q2_0p65_sigma_ct.png"

    _plot_sigma_ct(_measurement(), output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def _sigma_u_w_measurement() -> Measurement:
    q2_values = []
    w_values = []
    theta_centers = []
    for q2_center in (1.0, 1.55, 2.05, 2.55):
        row_w = (
            (1.65, 1.75, 1.85, 1.95, 2.05, 2.15)
            if q2_center == 2.55
            else (1.65, 1.75, 1.85, 1.95, 2.05, 2.15, 2.25, 2.35)
        )
        for w_center in row_w:
            for cosine_center in (-0.25, 0.35):
                q2_values.append(q2_center)
                w_values.append(w_center)
                theta_centers.append(np.arccos(cosine_center))

    points = len(q2_values)
    return Measurement(
        xsect=Xsect(
            sigU=np.linspace(5.0, 100.0, points),
            sigU_err=np.full(points, 2.0),
        ),
        kinematics=Kinematics(
            Q2=np.asarray(q2_values),
            W=np.asarray(w_values),
            thetacm_center=np.asarray(theta_centers),
        ),
        metadata=Metadata(
            reaction="E- P --> E- K+ LAMBDA X",
            unit="nb/sr",
            angle_unit="rad",
        ),
    )


def test_plot_sigmaU_W_writes_png(tmp_path: Path):
    output = tmp_path / "lambda_sigmaU_w.png"

    _plot_sigmaU_W(_sigma_u_w_measurement(), output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
