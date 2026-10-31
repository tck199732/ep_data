import json
import math
from pathlib import Path

import pytest

from ep_data.config import load_config
from ep_data.parser.clas_ep_2007 import parse

RAW_DATA = Path("data/hep/HEPData-ins732363-v1-yaml")
EXPECTED_LENGTHS = {
    "lambda_e2p567_fine_w.json": 90,
    "lambda_e4.json": 180,
    "lambda_e2p567_coarse_w.json": 58,
    "sigma0_e2p567_fine_w.json": 83,
    "sigma0_e4.json": 113,
    "sigma0_e2p567_coarse_w.json": 47,
    "lambda_simultaneous_epsilon_phi.json": 24,
    "lambda_rosenbluth.json": 24,
    "sigma0_simultaneous_epsilon_phi.json": 18,
    "sigma0_rosenbluth.json": 18,
}


def test_clas_dataset_uses_approved_parser():
    config = load_config(Path("data/hep_config.json"))

    assert config["clas_ep_2007"].parser == "clas_ep_2007"


@pytest.mark.skipif(
    not RAW_DATA.is_dir(), reason="downloaded submission is unavailable"
)
def test_parser_writes_ten_homogeneous_response_measurements(tmp_path):
    parse(RAW_DATA, tmp_path)

    assert {path.name for path in tmp_path.glob("*.json")} == set(EXPECTED_LENGTHS)
    measurements = {
        path.name: json.loads(path.read_text()) for path in tmp_path.glob("*.json")
    }
    assert {
        name: len(measurement["kinematics"]["Q2"])
        for name, measurement in measurements.items()
    } == EXPECTED_LENGTHS

    first = measurements["lambda_e2p567_fine_w.json"]
    assert first["xsect"]["sigU"][0] == pytest.approx(82.46)
    assert first["xsect"]["sigU_err"][0] == pytest.approx(6.13)
    assert first["xsect"]["sigTT"][0] == pytest.approx(1.55)
    assert first["xsect"]["sigLT"][0] == pytest.approx(2.16)
    assert first["kinematics"]["Q2"][0] == pytest.approx(0.65)
    assert first["kinematics"]["W"][0] == pytest.approx(1.65)
    assert math.isnan(first["kinematics"]["thetacm"][0])
    assert first["kinematics"]["thetacm_center"][0] == pytest.approx(math.acos(-0.6))
    assert first["kinematics"]["epsilon"] is None

    metadata = first["metadata"]
    assert metadata["dataset_key"] == "clas_ep_2007"
    assert metadata["experiment"] == "CLAS"
    assert metadata["year"] == 2007
    assert metadata["unit"] == "nb/sr"
    assert metadata["source_unit"] == "NB/STR"
    assert metadata["angle_unit"] == "rad"
    assert "Phys. Rev. C 75, 045203 (2007)" in metadata["reference"]
    assert any(
        qualifier["name"] == "COS(THETA) BINS" for qualifier in metadata["qualifiers"]
    )

    sigma_four = measurements["sigma0_e4.json"]
    assert sigma_four["metadata"]["table_id"] == "Tables 70-76, Tables 84-95"


@pytest.mark.skipif(
    not RAW_DATA.is_dir(), reason="downloaded submission is unavailable"
)
def test_parser_maps_missing_sentinels_and_separated_errors(tmp_path):
    parse(RAW_DATA, tmp_path)

    lambda_four = json.loads((tmp_path / "lambda_e4.json").read_text())
    assert any(math.isnan(value) for value in lambda_four["xsect"]["sigU"])
    assert any(math.isnan(value) for value in lambda_four["xsect"]["sigTT"])
    assert any(math.isnan(value) for value in lambda_four["xsect"]["sigLT"])
    for component, sentinels in {
        "sigU": {0.0},
        "sigTT": {0.0, 1.0},
        "sigLT": {0.0, 1.41},
    }.items():
        values = lambda_four["xsect"][component]
        errors = lambda_four["xsect"][f"{component}_err"]
        assert not any(
            value in sentinels and error == 0.0
            for value, error in zip(values, errors, strict=True)
        )

    simultaneous = json.loads(
        (tmp_path / "lambda_simultaneous_epsilon_phi.json").read_text()
    )
    assert simultaneous["xsect"]["sigL"][0] == pytest.approx(-11.07)
    assert simultaneous["xsect"]["sigL_err"][0] == pytest.approx(27.44)
    assert simultaneous["xsect"]["sigL_sys_err_low"][0] == pytest.approx(46.34)
    assert simultaneous["xsect"]["sigL_sys_err_high"][0] == pytest.approx(46.34)
    assert simultaneous["xsect"]["sigTT_sys_err_low"][0] == 0.0

    rosenbluth = json.loads((tmp_path / "lambda_rosenbluth.json").read_text())
    assert rosenbluth["xsect"]["sigL"][0] == pytest.approx(-43.11)
    assert rosenbluth["xsect"]["sigT"][0] == pytest.approx(100.53)
    assert rosenbluth["xsect"]["sigTT"] is None
    assert "SIG(C=L)/SIG(C=T)" not in rosenbluth["metadata"]["observable"]
