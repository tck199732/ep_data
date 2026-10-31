import json
import math
from pathlib import Path

import pytest

from ep_data.config import load_config
from ep_data.parser.bonn_yp_1970 import parse
from ep_data.parser.physics import MASS_PROTON

RAW_DATA = Path("data/hep/HEPData-ins63868-v1-yaml")
EXPECTED_OUTPUTS = {
    "lambda_1p3gev.json",
    "lambda_1p327_1p344gev.json",
    "sigma0_1p45gev.json",
    "sigma0_1p410_1p416gev.json",
}


def test_bonn_dataset_is_registered_under_approved_name():
    config = load_config(Path("data/hep_config.json"))

    assert "bonn_yp_1970" in config
    assert "desy_yp_1970" not in config
    assert config["bonn_yp_1970"].parser == "bonn_yp_1970"


@pytest.mark.skipif(
    not RAW_DATA.is_dir(), reason="downloaded submission is unavailable"
)
def test_parser_writes_four_pointwise_bonn_measurements(tmp_path):
    parse(RAW_DATA, tmp_path)

    assert {path.name for path in tmp_path.glob("*.json")} == EXPECTED_OUTPUTS
    measurements = {
        path.name: json.loads(path.read_text()) for path in tmp_path.glob("*.json")
    }

    fixed_lambda = measurements["lambda_1p3gev.json"]
    assert fixed_lambda["xsect"]["sigU"][0] == pytest.approx(321.0)
    assert fixed_lambda["xsect"]["sigU_err"][0] == pytest.approx(33.0)
    assert fixed_lambda["xsect"]["sigU_sys_err_low"][0] == pytest.approx(16.05)
    assert fixed_lambda["xsect"]["sigU_sys_err_high"][0] == pytest.approx(16.05)
    assert fixed_lambda["kinematics"]["Q2"] == [0.0] * 11
    assert fixed_lambda["kinematics"]["W"][0] == pytest.approx(
        math.sqrt(MASS_PROTON**2 + 2.0 * MASS_PROTON * 1.3)
    )
    assert fixed_lambda["kinematics"]["thetacm"] == pytest.approx(
        [
            math.radians(value)
            for value in [
                6.0,
                10.0,
                15.0,
                20.0,
                30.0,
                40.0,
                50.0,
                60.0,
                70.0,
                80.0,
                90.0,
            ]
        ]
    )
    assert fixed_lambda["kinematics"]["thetacm_center"] is None
    assert "Egamma" not in fixed_lambda["kinematics"]

    metadata = fixed_lambda["metadata"]
    assert metadata["dataset_key"] == "bonn_yp_1970"
    assert metadata["experiment"] == "Bonn"
    assert metadata["year"] == 1970
    assert metadata["unit"] == "nb/sr"
    assert metadata["source_unit"] == "MUB/SR"
    assert metadata["angle_unit"] == "rad"
    assert metadata["observable"] == "D(SIG)/DOMEGA"
    observable = next(
        qualifier
        for qualifier in metadata["qualifiers"]
        if qualifier["name"] == "OBSERVABLE"
    )
    assert observable == {
        "name": "OBSERVABLE",
        "value": "D(SIG)/DOMEGA",
        "unit": "nb/sr",
    }

    pointwise_lambda = measurements["lambda_1p327_1p344gev.json"]
    expected_energies = [1.344, 1.343, 1.342, 1.34, 1.336, 1.332, 1.327]
    expected_w = [
        math.sqrt(MASS_PROTON**2 + 2.0 * MASS_PROTON * energy)
        for energy in expected_energies
    ]
    assert pointwise_lambda["kinematics"]["W"] == pytest.approx(expected_w)
    assert pointwise_lambda["kinematics"]["Q2"] == [0.0] * 7
    assert pointwise_lambda["kinematics"]["thetacm"] == pytest.approx(
        [math.radians(value) for value in [11.2, 17.7, 23.3, 33.5, 43.2, 52.7, 62.0]]
    )

    assert measurements["sigma0_1p45gev.json"]["metadata"]["reaction"] == (
        "GAMMA P --> K+ SIGMA0"
    )
    assert len(measurements["sigma0_1p410_1p416gev.json"]["xsect"]["sigU"]) == 3
