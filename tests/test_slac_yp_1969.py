import json
from pathlib import Path

import numpy as np
import pytest

from ep_data.parser.physics import (
    MASS_LAMBDA,
    MASS_SIGMA0,
    get_dOmega_dt,
    get_dt_dOmega,
)
from ep_data.parser.slac_yp_1969 import parse

RAW_DATA = Path("data/hep/HEPData-ins54849-v1-yaml")
EXPECTED_OUTPUTS = {
    "lambda_5gev.json",
    "lambda_8gev.json",
    "lambda_11gev.json",
    "lambda_16gev.json",
    "sigma0_5gev.json",
    "sigma0_8gev.json",
    "sigma0_11gev.json",
    "sigma0_16gev.json",
}


def test_hyperon_mass_controls_jacobian_and_default_is_lambda():
    q2 = np.array([0.0])
    w = np.array([3.203])

    default = get_dt_dOmega(q2, w)
    lambda_result = get_dt_dOmega(q2, w, MASS_LAMBDA)
    sigma_result = get_dt_dOmega(q2, w, MASS_SIGMA0)

    np.testing.assert_allclose(default, lambda_result)
    assert sigma_result[0] < lambda_result[0]
    np.testing.assert_allclose(
        get_dOmega_dt(q2, w, MASS_SIGMA0) * sigma_result,
        np.ones(1),
    )


@pytest.mark.skipif(
    not RAW_DATA.is_dir(), reason="downloaded submission is unavailable"
)
def test_parser_converts_units_and_excludes_table5(tmp_path):
    parse(RAW_DATA, tmp_path)

    assert {path.name for path in tmp_path.glob("*.json")} == EXPECTED_OUTPUTS

    lambda_5 = json.loads((tmp_path / "lambda_5gev.json").read_text())
    lambda_factor = float(
        get_dt_dOmega(np.array([0.0]), np.array([3.203]), MASS_LAMBDA)[0]
    )
    assert lambda_5["xsect"]["sigU"][0] == pytest.approx(
        0.00034 * 1.0e6 * lambda_factor
    )
    assert lambda_5["xsect"]["sigU_err"][0] == pytest.approx(
        0.00006 * 1.0e6 * lambda_factor
    )
    assert lambda_5["xsect"]["sigU_sys_err_low"][0] == pytest.approx(
        lambda_5["xsect"]["sigU"][0] * 0.10
    )
    assert lambda_5["kinematics"]["Q2"] == [0.0] * 14
    assert lambda_5["kinematics"]["t"][0] == 0.012
    assert lambda_5["kinematics"]["t_center"] is None
    assert lambda_5["metadata"]["unit"] == "nb/sr"
    assert lambda_5["metadata"]["source_unit"] == "MB/GEV**2"
    assert lambda_5["metadata"]["observable"] == "D(SIG)/DOMEGA"
    assert {
        (item["name"], item["value"], item["unit"])
        for item in lambda_5["metadata"]["qualifiers"]
        if item["name"] == "OBSERVABLE"
    } == {("OBSERVABLE", "D(SIG)/DOMEGA", "nb/sr")}

    sigma_5 = json.loads((tmp_path / "sigma0_5gev.json").read_text())
    sigma_factor = float(
        get_dt_dOmega(np.array([0.0]), np.array([3.203]), MASS_SIGMA0)[0]
    )
    assert sigma_5["xsect"]["sigU"][0] == pytest.approx(0.000173 * 1.0e6 * sigma_factor)
    assert sigma_factor != pytest.approx(lambda_factor)
