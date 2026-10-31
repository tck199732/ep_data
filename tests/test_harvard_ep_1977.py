import json
import math
from pathlib import Path

import pytest

from ep_data.parser.harvard_ep_1977 import parse

RAW_DATA = Path("data/hep/HEPData-ins110317-v1-yaml")
EXPECTED_OUTPUTS = {
    "cornell_lambda.json",
    "harvard_cornell_lambda.json",
    "cea_lambda.json",
    "cornell_sigma0.json",
    "harvard_cornell_sigma0.json",
    "cea_sigma0.json",
    "cornell_deuterium_sigma0.json",
    "cornell_deuterium_sigma_minus.json",
}


@pytest.mark.skipif(
    not RAW_DATA.is_dir(), reason="downloaded submission is unavailable"
)
def test_parser_writes_one_file_per_homogeneous_measurement(tmp_path):
    parse(RAW_DATA, tmp_path)

    assert {path.name for path in tmp_path.glob("*.json")} == EXPECTED_OUTPUTS
    measurements = {
        path.name: json.loads(path.read_text()) for path in tmp_path.glob("*.json")
    }
    assert {item["metadata"]["experiment"] for item in measurements.values()} == {
        "Cornell",
        "Harvard-Cornell",
        "CEA",
    }
    assert {
        name: len(item["xsect"]["sigU"]) for name, item in measurements.items()
    } == {
        "cornell_lambda.json": 11,
        "harvard_cornell_lambda.json": 4,
        "cea_lambda.json": 8,
        "cornell_sigma0.json": 11,
        "harvard_cornell_sigma0.json": 4,
        "cea_sigma0.json": 8,
        "cornell_deuterium_sigma0.json": 6,
        "cornell_deuterium_sigma_minus.json": 6,
    }
    assert {item["metadata"]["observable"] for item in measurements.values()} == {
        "D(SIG(Q=LAM))/DOMEGA",
        "D(SIG(Q=SIG))/DOMEGA",
        "D(SIG(Q=SIG0))/DOMEGA",
        "D(SIG(Q=SIG-))/DOMEGA",
    }
    assert all(item["metadata"]["unit"] == "nb/sr" for item in measurements.values())

    harvard_lambda = measurements["harvard_cornell_lambda.json"]
    assert math.isnan(harvard_lambda["kinematics"]["thetacm"][2])
    assert harvard_lambda["kinematics"]["thetacm_center"][2] == pytest.approx(
        math.radians(7.5)
    )
    assert harvard_lambda["metadata"]["angle_unit"] == "rad"

    cea_sigma = measurements["cea_sigma0.json"]
    assert math.isnan(cea_sigma["xsect"]["sigU"][0])
    assert cea_sigma["xsect"]["sigU_upper_limit"][0] == 53.0
    assert math.isnan(cea_sigma["xsect"]["sigU_sys_err_low"][0])
    assert math.isnan(cea_sigma["xsect"]["sigU"][4])
    assert cea_sigma["xsect"]["sigU_upper_limit"][4] == 42.0

    table3_sigma0 = measurements["cornell_deuterium_sigma0.json"]
    table3_sigma_minus = measurements["cornell_deuterium_sigma_minus.json"]
    assert table3_sigma0["metadata"]["reaction"] == "GAMMA* P --> K+ SIGMA0"
    assert table3_sigma_minus["metadata"]["reaction"] == "GAMMA* N --> K+ SIGMA-"
    assert any(
        qualifier["name"] == "RE (Q=DEUT)"
        for qualifier in table3_sigma_minus["metadata"]["qualifiers"]
    )
