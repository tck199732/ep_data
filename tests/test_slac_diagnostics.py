from pathlib import Path

import numpy as np

from ep_data.diagnostics.plots.slac_yp_1969 import (
    _combined_uncertainty,
    _plot_sigmaU_t,
    _to_dsigma_dt,
)
from ep_data.models import Kinematics, Measurement, Metadata, Xsect
from ep_data.parser.physics import MASS_LAMBDA, get_dOmega_dt


def _measurement(energy: int) -> Measurement:
    scale = 20.0 / energy
    return Measurement(
        xsect=Xsect(
            sigU=np.asarray((100.0, 30.0, 10.0)) * scale,
            sigU_err=np.asarray((3.0, 2.0, 1.0)),
            sigU_sys_err_low=np.asarray((4.0, 1.5, 0.5)),
            sigU_sys_err_high=np.asarray((4.0, 1.5, 0.5)),
        ),
        kinematics=Kinematics(
            Q2=np.zeros(3),
            W=np.full(3, 3.0),
            t=np.asarray((0.01, 0.5, 2.0)),
        ),
        metadata=Metadata(
            reaction="GAMMA P --> K+ LAMBDA",
            unit="nb/sr",
        ),
    )


def test_combined_uncertainty_adds_statistical_and_systematic_in_quadrature():
    combined = _combined_uncertainty(_measurement(5))

    expected = np.asarray((5.0, 2.5, np.hypot(1.0, 0.5)))
    np.testing.assert_allclose(combined[0], expected)
    np.testing.assert_allclose(combined[1], expected)


def test_plot_sigmaU_t_writes_png(tmp_path: Path):
    measurements = {
        f"lambda_{energy}gev": _measurement(energy) for energy in (5, 8, 11, 16)
    }
    output = tmp_path / "lambda_sigmaU_t.png"

    _plot_sigmaU_t(measurements, output)

    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_to_dsigma_dt_applies_inverse_jacobian_and_microbarn_conversion():
    measurement = _measurement(5)

    converted, uncertainty = _to_dsigma_dt(measurement)

    factor = (
        get_dOmega_dt(
            measurement.kinematics.Q2,
            measurement.kinematics.W,
            MASS_LAMBDA,
        )
        * 1.0e-3
    )
    np.testing.assert_allclose(converted, measurement.xsect.sigU * factor)
    np.testing.assert_allclose(
        uncertainty,
        _combined_uncertainty(measurement) * factor[np.newaxis, :],
    )
