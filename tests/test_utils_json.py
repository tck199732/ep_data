import json
from pathlib import Path

import numpy as np
import pytest

from ep_data.models import (
    Kinematics,
    Measurement,
    Metadata,
    Qualifier,
    Xsect,
)
from ep_data.utils import jsonify, write_json_atomic


def make_measurement():
    return Measurement(
        xsect=Xsect(sigU=np.array([1.0, 2.0])),
        kinematics=Kinematics(Q2=[0.5, 1.0], W=[2.0, 2.1]),
        metadata=Metadata(
            reaction="GAMMA* P --> K+ LAMBDA",
            qualifiers=(Qualifier("SQRT(S)", "2.0-2.1", "GeV"),),
        ),
    )


def test_jsonify_converts_measurement_recursively():
    converted = jsonify(make_measurement())

    assert converted["xsect"]["sigU"] == [1.0, 2.0]
    assert converted["xsect"]["sigU_err"] == [0.0, 0.0]
    assert converted["metadata"]["qualifiers"] == [
        {"name": "SQRT(S)", "value": "2.0-2.1", "unit": "GeV"}
    ]


def test_jsonify_handles_numpy_scalars_and_paths(tmp_path):
    converted = jsonify(
        {"integer": np.int64(3), "path": tmp_path, "items": (np.float64(1.5),)}
    )
    assert converted == {
        "integer": 3,
        "path": str(tmp_path),
        "items": [1.5],
    }


def test_jsonify_rejects_multidimensional_arrays():
    with pytest.raises(ValueError, match="one-dimensional"):
        jsonify(np.ones((2, 2)))


def test_atomic_json_writer_accepts_dataclass(tmp_path):
    path = tmp_path / "measurement.json"
    write_json_atomic(path, make_measurement())

    stored = json.loads(path.read_text())
    assert stored["metadata"]["reaction"] == "GAMMA* P --> K+ LAMBDA"
    assert stored["kinematics"]["Q2"] == [0.5, 1.0]
    assert Path(path).is_file()
